"""
CDR-Mamba backbone with 4-module ablation switches (M1-M4).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel

from models.module_switches import resolve_module_plan


def _build_resnet18_imagenet():
    # torchvision>=0.13 uses `weights`; older versions use `pretrained`.
    try:
        if hasattr(models, "ResNet18_Weights"):
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        return models.resnet18(pretrained=True)
    except Exception:
        # Offline-safe fallback. Checkpoint loading will overwrite these params.
        if hasattr(models, "ResNet18_Weights"):
            return models.resnet18(weights=None)
        return models.resnet18(pretrained=False)


def js_divergence(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def canonical_fusion_mode(name: str) -> str:
    mode = str(name or "full").lower()
    alias = {
        "w/o_alpha": "wo_alpha",
        "no_alpha": "wo_alpha",
        "w/o_rho": "wo_rho",
        "w/o_r": "wo_rho",
        "w/o_reliability": "wo_rho",
        "no_rho": "wo_rho",
        "no_reliability": "wo_rho",
        "concat+mlp": "concat_mlp",
        "concat": "concat_mlp",
    }
    mode = alias.get(mode, mode)
    valid = {"full", "wo_alpha", "wo_rho", "sum", "concat_mlp", "simple"}
    if mode not in valid:
        raise ValueError(f"Unsupported cdr_fusion_mode={name}. Expected one of {sorted(valid)}.")
    return mode


def _sinusoidal_positional_encoding(length: int, dim: int, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.to(dtype=dtype).unsqueeze(0)


class SelectiveStateBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.state_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        v = self.in_proj(x)
        g = torch.sigmoid(self.gate_proj(x))
        v = self.state_conv(v.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(g * v)
        x = self.dropout(x)
        return residual + x


class BiMamba2Encoder(nn.Module):
    def __init__(self, d_model, num_layers=2, dropout=0.1):
        super().__init__()
        self.fwd_layers = nn.ModuleList(
            [SelectiveStateBlock(d_model=d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [SelectiveStateBlock(d_model=d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_f = x
        x_b = torch.flip(x, dims=[1])
        for layer in self.fwd_layers:
            x_f = layer(x_f)
        for layer in self.bwd_layers:
            x_b = layer(x_b)
        x_b = torch.flip(x_b, dims=[1])
        return self.norm(0.5 * (x_f + x_b))


class BiTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers=2,
        dropout=0.1,
        nhead=8,
        dim_ffn=2048,
        max_len=512,
    ):
        super().__init__()
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        layer_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            activation="gelu",
        )
        self.batch_first = True
        try:
            encoder_layer = nn.TransformerEncoderLayer(batch_first=True, norm_first=True, **layer_kwargs)
        except TypeError:
            self.batch_first = False
            encoder_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        n = x.size(1)
        if n <= self.max_len:
            x = x + self.pos_embed[:, :n, :]
        else:
            x = x + _sinusoidal_positional_encoding(n, x.size(-1), x.device, x.dtype)
        x = self.drop(x)
        if self.batch_first:
            x = self.encoder(x)
        else:
            x = self.encoder(x.transpose(0, 1)).transpose(0, 1)
        return self.norm(x)


class VSSDEncoder(nn.Module):
    def __init__(self, d_model, num_layers=2, dropout=0.1, encoder_builder=None):
        super().__init__()
        resnet = _build_resnet18_imagenet()
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.patch_proj = nn.Conv2d(512, d_model, kernel_size=1, bias=False)
        if encoder_builder is None:
            self.encoder = BiMamba2Encoder(d_model=d_model, num_layers=num_layers, dropout=dropout)
        else:
            self.encoder = encoder_builder(num_layers=num_layers)

    def forward(self, image):
        feat_map = self.stem(image)
        patch_tokens = self.patch_proj(feat_map).flatten(2).transpose(1, 2)
        return self.encoder(patch_tokens)


class EmotionSlotRouter(nn.Module):
    def __init__(self, d_model, num_slots, tau):
        super().__init__()
        self.tau = tau
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

    def forward(self, token_states):
        dist2 = torch.sum(
            (token_states.unsqueeze(2) - self.slots.unsqueeze(0).unsqueeze(0)) ** 2,
            dim=-1,
        )  # [B, N, M]
        route = torch.softmax(-dist2 / self.tau, dim=-1)
        slot_states = torch.einsum("bnm,bnd->bmd", route, token_states)
        return slot_states, route


class CDRMamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.d_model = args.cdr_d_model
        self.num_slots = args.cdr_num_slots
        self.tau_route = args.cdr_tau_route
        self.tau_consensus = args.cdr_tau_consensus
        self.eps = 1e-8

        self.module_plan = resolve_module_plan(
            ablation=getattr(args, "ablation", "none"),
            module_combo=getattr(args, "cdr_module_combo", "from_ablation"),
        )
        self.ablation = self.module_plan["ablation"]
        self.module_combo = self.module_plan["module_combo"]
        self.module_notes = list(self.module_plan["notes"])
        self.active_modules = dict(self.module_plan["active"])
        self.requested_modules = dict(self.module_plan["requested"])
        self.cdr_consensus_only = self.ablation == "m7_consensus_only"
        self.cdr_divergence_only = self.ablation == "m8_divergence_only"
        if self.cdr_consensus_only and self.cdr_divergence_only:
            raise ValueError("Invalid CDR variant: both consensus-only and divergence-only are enabled.")
        if self.cdr_consensus_only:
            self.cdr_internal_variant = "consensus_only"
        elif self.cdr_divergence_only:
            self.cdr_internal_variant = "divergence_only"
        else:
            self.cdr_internal_variant = "full"

        self.sequence_backbone = str(getattr(args, "cdr_backbone", "mamba")).lower()
        if self.sequence_backbone not in {"mamba", "transformer"}:
            raise ValueError("cdr_backbone must be one of {'mamba','transformer'}.")
        self.transformer_heads = int(getattr(args, "cdr_transformer_heads", 8))
        self.transformer_ffn_mult = float(getattr(args, "cdr_transformer_ffn_mult", 4.0))
        self.transformer_max_len = int(getattr(args, "cdr_transformer_max_len", 512))

        self.fusion_mode = canonical_fusion_mode(getattr(args, "cdr_fusion_mode", "full"))
        self.fixed_alpha = float(getattr(args, "cdr_fixed_alpha", 0.5))
        self.fixed_r_t = float(getattr(args, "cdr_fixed_r_t", 0.5))
        self.fixed_r_v = float(getattr(args, "cdr_fixed_r_v", 0.5))
        self.concat_hidden = int(getattr(args, "cdr_concat_hidden", self.d_model * 2))

        self.sasp_replacement = str(getattr(args, "cdr_sasp_replacement", "none")).lower()
        self.allow_raw_cdr = bool(getattr(args, "cdr_allow_raw_cdr", False))

        self.disable_mamba = not self.active_modules["m1"]
        self.disable_slot = not self.active_modules["m2"]
        self.disable_cdr = not self.active_modules["m3"]
        self.disable_reconcile = not self.active_modules["m3"]
        if self.disable_slot and self.allow_raw_cdr and self.requested_modules.get("m3", False):
            self.disable_cdr = False
            self.disable_reconcile = False
            self.module_notes.append("raw_cdr_enabled_without_sasp")
        self.use_raw_pool_replacement = self.disable_slot and self.allow_raw_cdr and (self.sasp_replacement == "raw_pool")

        self.text_model = BertModel.from_pretrained(args.bert_model_path)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, self.d_model)

        def _make_seq_encoder(num_layers):
            if self.sequence_backbone == "transformer":
                dim_ffn = int(self.transformer_ffn_mult * self.d_model)
                return BiTransformerEncoder(
                    d_model=self.d_model,
                    num_layers=num_layers,
                    dropout=args.cdr_dropout,
                    nhead=self.transformer_heads,
                    dim_ffn=dim_ffn,
                    max_len=self.transformer_max_len,
                )
            return BiMamba2Encoder(
                d_model=self.d_model,
                num_layers=num_layers,
                dropout=args.cdr_dropout,
            )

        # ===== Module-1: Mamba Backbone =====
        if not self.disable_mamba:
            self.text_encoder = _make_seq_encoder(num_layers=args.cdr_text_layers)
            self.vision_encoder = VSSDEncoder(
                d_model=self.d_model,
                num_layers=args.cdr_vision_layers,
                dropout=args.cdr_dropout,
                encoder_builder=_make_seq_encoder,
            )
            self.vision_stem = None
            self.vision_patch_proj = None
        else:
            self.text_encoder = None
            self.vision_encoder = None
            resnet = _build_resnet18_imagenet()
            self.vision_stem = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            self.vision_patch_proj = nn.Conv2d(512, self.d_model, kernel_size=1, bias=False)

        # ===== Module-3 state stream encoders =====
        if not self.disable_cdr:
            self.consensus_state = _make_seq_encoder(num_layers=args.cdr_state_layers)
            self.div_t_state = _make_seq_encoder(num_layers=args.cdr_state_layers)
            self.div_v_state = _make_seq_encoder(num_layers=args.cdr_state_layers)
        else:
            self.consensus_state = None
            self.div_t_state = None
            self.div_v_state = None

        # ===== Module-2: Shared Affective State Projection =====
        self.use_shared_affective_projection = not self.disable_slot
        self.router = (
            EmotionSlotRouter(d_model=self.d_model, num_slots=self.num_slots, tau=self.tau_route)
            if self.use_shared_affective_projection
            else None
        )

        # ===== Module-3: CDR Modeling + Reconciliation Decision =====
        self.use_cd_state_modeling = not self.disable_cdr

        self.use_reliability_reconciliation = not self.disable_reconcile
        self.alpha_mlp = (
            nn.Sequential(
                nn.Linear(3, args.cdr_gate_hidden),
                nn.ReLU(),
                nn.Linear(args.cdr_gate_hidden, 1),
            )
            if self.use_reliability_reconciliation
            else None
        )
        self.concat_fusion_mlp = (
            nn.Sequential(
                nn.Linear(self.d_model * 3, self.concat_hidden),
                nn.ReLU(),
                nn.Dropout(args.cdr_dropout),
                nn.Linear(self.concat_hidden, self.d_model),
            )
            if (self.use_reliability_reconciliation and self.fusion_mode == "concat_mlp")
            else None
        )

        # Backward-compatible aliases for trainer logs.
        self.use_slot_routing = self.use_shared_affective_projection
        self.use_cdr_modeling = self.use_cd_state_modeling
        self.use_reconciliation = self.use_reliability_reconciliation
        self.use_noise_robust_trainer = self.active_modules["m4"]

        self.text_head = nn.Linear(self.d_model, self.num_classes)
        self.vision_head = nn.Linear(self.d_model, self.num_classes)
        self.consensus_head = nn.Linear(self.d_model, self.num_classes)
        self.final_head = nn.Linear(self.d_model, self.num_classes)

    def _masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return torch.sum(x * m, dim=1) / m.sum(dim=1).clamp_min(1.0)

    def _normalized_confidence(self, probs):
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(self.eps)), dim=-1)
        return 1.0 - entropy / math.log(self.num_classes)

    def _encode_text(self, text):
        # Module-1 entry (text side).
        text_out = self.text_model(**text).last_hidden_state
        text_tokens = self.text_proj(text_out)
        if self.text_encoder is None:
            return text_tokens
        return self.text_encoder(text_tokens)

    def _encode_vision(self, image):
        # Module-1 entry (vision side).
        if self.vision_encoder is not None:
            return self.vision_encoder(image)
        feat_map = self.vision_stem(image)
        return self.vision_patch_proj(feat_map).flatten(2).transpose(1, 2)

    def _shared_affective_state_projection(self, text_tokens, vision_tokens):
        # Module-2 forward.
        if self.use_shared_affective_projection:
            z_t, route_t = self.router(text_tokens)
            z_v, route_v = self.router(vision_tokens)
        elif self.use_raw_pool_replacement:
            z_t = F.adaptive_avg_pool1d(text_tokens.transpose(1, 2), self.num_slots).transpose(1, 2)
            z_v = F.adaptive_avg_pool1d(vision_tokens.transpose(1, 2), self.num_slots).transpose(1, 2)
            route_t = None
            route_v = None
        else:
            # Strict deletion path for Module-2: no surrogate alignment is used.
            z_t = None
            z_v = None
            route_t = None
            route_v = None
        return z_t, z_v, route_t, route_v

    def _consensus_divergence_state_modeling(self, z_t, z_v):
        # Module-3 (CD modeling) forward.
        if (not self.use_cd_state_modeling) or (z_t is None) or (z_v is None):
            return {
                "slot_dist2": None,
                "g": None,
                "rho": None,
                "z_c": None,
                "z_dt": None,
                "z_dv": None,
                "h_c": None,
                "h_dt": None,
                "h_dv": None,
            }

        slot_dist2 = torch.sum((z_t - z_v) ** 2, dim=-1)
        g = torch.exp(-slot_dist2 / (2.0 * self.tau_consensus))
        rho = torch.mean(torch.norm(z_t - z_v, dim=-1), dim=-1)

        z_c = g.unsqueeze(-1) * ((z_t + z_v) * 0.5)
        z_dt = (1.0 - g).unsqueeze(-1) * z_t
        z_dv = (1.0 - g).unsqueeze(-1) * z_v

        if self.cdr_consensus_only:
            z_dt = torch.zeros_like(z_dt)
            z_dv = torch.zeros_like(z_dv)
        elif self.cdr_divergence_only:
            z_c = torch.zeros_like(z_c)

        if self.consensus_state is not None:
            if self.cdr_consensus_only:
                h_c = self.consensus_state(z_c).mean(dim=1)
                h_dt = torch.zeros_like(h_c)
                h_dv = torch.zeros_like(h_c)
            elif self.cdr_divergence_only:
                h_dt = self.div_t_state(z_dt).mean(dim=1)
                h_dv = self.div_v_state(z_dv).mean(dim=1)
                h_c = torch.zeros_like(h_dt)
            else:
                h_c = self.consensus_state(z_c).mean(dim=1)
                h_dt = self.div_t_state(z_dt).mean(dim=1)
                h_dv = self.div_v_state(z_dv).mean(dim=1)
        else:
            if self.cdr_consensus_only:
                h_c = z_c.mean(dim=1)
                h_dt = torch.zeros_like(h_c)
                h_dv = torch.zeros_like(h_c)
            elif self.cdr_divergence_only:
                h_dt = z_dt.mean(dim=1)
                h_dv = z_dv.mean(dim=1)
                h_c = torch.zeros_like(h_dt)
            else:
                h_c = z_c.mean(dim=1)
                h_dt = z_dt.mean(dim=1)
                h_dv = z_dv.mean(dim=1)

        return {
            "slot_dist2": slot_dist2,
            "g": g,
            "rho": rho,
            "z_c": z_c,
            "z_dt": z_dt,
            "z_dv": z_dv,
            "h_c": h_c,
            "h_dt": h_dt,
            "h_dv": h_dv,
        }

    def _reliability_guided_reconciliation(self, h_t, h_v, h_c, h_dt, h_dv):
        # Always produce unimodal predictions.
        pre_t = self.text_head(h_t)
        pre_v = self.vision_head(h_v)
        p_t = torch.softmax(pre_t, dim=-1)
        p_v = torch.softmax(pre_v, dim=-1)

        delta = js_divergence(p_t, p_v)
        q_t = self._normalized_confidence(p_t)
        q_v = self._normalized_confidence(p_v)
        q_sum = (q_t + q_v).clamp_min(self.eps)
        r_t = q_t / q_sum
        r_v = q_v / q_sum

        has_consensus = (h_c is not None) and (not self.cdr_divergence_only)
        if self.use_reliability_reconciliation and (h_c is not None):
            alpha_dyn = torch.sigmoid(self.alpha_mlp(torch.stack([delta, q_t, q_v], dim=-1))).squeeze(-1)
            alpha = alpha_dyn
            if self.fusion_mode == "wo_alpha":
                alpha = torch.full_like(alpha_dyn, self.fixed_alpha)

            if self.fusion_mode == "wo_rho":
                r_t_fuse = torch.full_like(r_t, self.fixed_r_t)
                r_v_fuse = torch.full_like(r_v, self.fixed_r_v)
                denom = (r_t_fuse + r_v_fuse).clamp_min(self.eps)
                r_t_fuse = r_t_fuse / denom
                r_v_fuse = r_v_fuse / denom
            elif self.fusion_mode in {"sum", "simple", "concat_mlp"}:
                r_t_fuse = torch.full_like(r_t, 0.5)
                r_v_fuse = torch.full_like(r_v, 0.5)
            else:
                r_t_fuse = r_t
                r_v_fuse = r_v

            if self.fusion_mode in {"sum", "simple"}:
                if self.cdr_consensus_only:
                    h_f = h_c
                elif self.cdr_divergence_only:
                    h_f = h_dt + h_dv
                else:
                    h_f = h_c + h_dt + h_dv
                alpha = torch.zeros_like(alpha_dyn)
            elif self.fusion_mode == "concat_mlp":
                if self.concat_fusion_mlp is None:
                    raise RuntimeError("concat_mlp fusion requested but concat_fusion_mlp is not initialized.")
                h_stack = torch.cat([h_c, h_dt, h_dv], dim=-1)
                h_f = self.concat_fusion_mlp(h_stack)
                alpha = torch.zeros_like(alpha_dyn)
            elif self.cdr_consensus_only:
                h_f = h_c
            elif self.cdr_divergence_only:
                h_f = alpha.unsqueeze(-1) * (
                    r_t_fuse.unsqueeze(-1) * h_dt + r_v_fuse.unsqueeze(-1) * h_dv
                )
            else:
                h_f = h_c + alpha.unsqueeze(-1) * (
                    r_t_fuse.unsqueeze(-1) * h_dt + r_v_fuse.unsqueeze(-1) * h_dv
                )

            if has_consensus:
                pre_c = self.consensus_head(h_c)
                p_c = torch.softmax(pre_c, dim=-1)
            else:
                pre_c = None
                p_c = None
            pre_f = self.final_head(h_f)
            p_f = torch.softmax(pre_f, dim=-1)
        else:
            alpha = torch.zeros_like(delta)
            h_f = 0.5 * (h_t + h_v)
            pre_c = None
            p_c = None
            pre_f = 0.5 * (pre_t + pre_v)
            p_f = torch.softmax(pre_f, dim=-1)

        return {
            "pre_t": pre_t,
            "pre_v": pre_v,
            "pre_c": pre_c,
            "pre_f": pre_f,
            "p_t": p_t,
            "p_v": p_v,
            "p_c": p_c,
            "p_f": p_f,
            "delta": delta,
            "q_t": q_t,
            "q_v": q_v,
            "alpha": alpha,
            "r_t": r_t_fuse if self.use_reliability_reconciliation and (h_c is not None) else r_t,
            "r_v": r_v_fuse if self.use_reliability_reconciliation and (h_c is not None) else r_v,
            "h_f": h_f,
        }

    def forward(self, image, text):
        text_tokens = self._encode_text(text)
        text_mask = text.get("attention_mask", None)
        h_t = self._masked_mean(text_tokens, text_mask)

        vision_tokens = self._encode_vision(image)
        h_v = vision_tokens.mean(dim=1)

        if self.use_shared_affective_projection or self.use_raw_pool_replacement:
            z_t, z_v, route_t, route_v = self._shared_affective_state_projection(
                text_tokens, vision_tokens
            )
        else:
            z_t, z_v, route_t, route_v = None, None, None, None

        cd_out = self._consensus_divergence_state_modeling(z_t, z_v)
        rec_out = self._reliability_guided_reconciliation(
            h_t=h_t,
            h_v=h_v,
            h_c=cd_out["h_c"],
            h_dt=cd_out["h_dt"],
            h_dv=cd_out["h_dv"],
        )

        return {
            "pre_t": rec_out["pre_t"],
            "pre_v": rec_out["pre_v"],
            "pre_c": rec_out["pre_c"],
            "pre_f": rec_out["pre_f"],
            "p_t": rec_out["p_t"],
            "p_v": rec_out["p_v"],
            "p_c": rec_out["p_c"],
            "p_f": rec_out["p_f"],
            "h_t": h_t,
            "h_v": h_v,
            "h_c": cd_out["h_c"],
            "h_dt": cd_out["h_dt"],
            "h_dv": cd_out["h_dv"],
            "delta": rec_out["delta"],
            "q_t": rec_out["q_t"],
            "q_v": rec_out["q_v"],
            "alpha": rec_out["alpha"],
            "r_t": rec_out["r_t"],
            "r_v": rec_out["r_v"],
            "rho": cd_out["rho"],
            "g": cd_out["g"],
            "z_t": z_t,
            "z_v": z_v,
            "z_c": cd_out["z_c"],
            "z_dt": cd_out["z_dt"],
            "z_dv": cd_out["z_dv"],
            "route_t": route_t,
            "route_v": route_v,
        }


class DMD(CDRMamba):
    pass
