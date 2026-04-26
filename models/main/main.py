import contextlib
import os
import pickle
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import BertTokenizer

from train_utils import AverageMeter, EMA
from .main_utils import Get_Scalar
from .cdr_method_helpers import (
    CDRMethodHelperMixin,
    gce_loss,
    js_divergence_prob,
    kl_divergence_prob,
    soft_ce_loss,
)
from models.module_switches import resolve_module_plan
from models.nets import dmd


class S2_VER(CDRMethodHelperMixin):
    def __init__(
        self,
        net_builder,
        num_classes,
        ema_m,
        T,
        p_cutoff,
        lambda_u,
        hard_label=True,
        t_fn=None,
        p_fn=None,
        it=0,
        tb_log=None,
        args=None,
        logger=None,
    ):
        super(S2_VER, self).__init__()
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.module_plan = resolve_module_plan(
            ablation=getattr(args, "ablation", "none"),
            module_combo=getattr(args, "cdr_module_combo", "from_ablation"),
        )
        self.ablation = self.module_plan["ablation"]
        self.module_combo = self.module_plan["module_combo"]
        self.active_modules = dict(self.module_plan["active"])
        self.requested_modules = dict(self.module_plan["requested"])
        self.module_notes = list(self.module_plan["notes"])

        # ------------------------------------------------------------------
        # Module-4: Noise-Robust Optimization Strategy (training-only module)
        # Scope:
        # - warmup GCE
        # - clean/noisy split + GMM score
        # - soft relabel (y_bar)
        # - EMA teacher target
        # - consistency loss (unsup)
        # - orth/align regularization coupling
        #
        # Manual comment control (direct deletion mode):
        # If you do not want CLI ablation, keep --ablation none and directly
        # edit this line to force False:
        # self.noise_robust_on = False
        # ------------------------------------------------------------------
        self.noise_robust_on = bool(self.active_modules["m4"])
        self.use_ema_teacher = self.noise_robust_on
        self.use_orth_loss = bool(self.active_modules["m5"])
        self.use_align_loss = bool(self.active_modules["m6"])

        self.model = dmd.DMD(args)
        self.ema_model = None
        self.ema = None

        self.t_fn = Get_Scalar(T)
        self.p_fn = Get_Scalar(p_cutoff)
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None
        self.it = it
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        self.image_norm_stats = {
            "mvsa-s": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "fi": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "se30k8": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }
        self._log_ablation_summary()

    def _log_ablation_summary(self):
        model = self.model
        active = self.active_modules
        summary = {
            "ablation": self.ablation,
            "module_combo": self.module_combo,
            "requested_modules": self.requested_modules,
            "active_modules": active,
            "module_notes": self.module_notes,
            "noise_robust_on": self.noise_robust_on,
            "use_ema_teacher": self.use_ema_teacher,
            "m1_mamba_backbone_on": active["m1"],
            "m2_shared_projection_on": active["m2"],
            "m3_cdr_decision_core_on": active["m3"],
            "m4_noise_robust_on": self.noise_robust_on,
            "m5_orth_loss_on": self.use_orth_loss,
            "m6_align_loss_on": self.use_align_loss,
            # 5-part method-aligned keys (for paper/module tracing)
            "module1_text_vision_state_encoder_on": active["m1"],
            "module2_shared_emotion_slot_router_on": getattr(model, "use_shared_affective_projection", None),
            "module3_consensus_divergence_state_modeling_on": getattr(model, "use_cd_state_modeling", None),
            "module4_reliability_guided_reconciliation_on": getattr(
                model, "use_reliability_reconciliation", None
            ),
            "cdr_internal_variant": getattr(model, "cdr_internal_variant", "full"),
            "cdr_consensus_only_on": getattr(model, "cdr_consensus_only", False),
            "cdr_divergence_only_on": getattr(model, "cdr_divergence_only", False),
            "module5_noise_robust_trainer_on": self.noise_robust_on,
            "module6_orth_loss_on": self.use_orth_loss,
            "module7_align_loss_on": self.use_align_loss,
            # backward-compatible aliases
            "module1_shared_projection_on": getattr(model, "use_shared_affective_projection", None),
            "module2_cd_state_modeling_on": getattr(model, "use_cd_state_modeling", None),
            "module3_reliability_reconcile_on": getattr(model, "use_reliability_reconciliation", None),
            "text_encoder": model.text_encoder.__class__.__name__ if hasattr(model, "text_encoder") else None,
            "vision_encoder": model.vision_encoder.__class__.__name__ if hasattr(model, "vision_encoder") else None,
            "cdr_backbone": getattr(model, "sequence_backbone", "mamba"),
            "cdr_fusion_mode": getattr(model, "fusion_mode", "full"),
            "cdr_use_raw_pool_replacement": getattr(model, "use_raw_pool_replacement", False),
            "cdr_sasp_replacement": getattr(model, "sasp_replacement", "none"),
        }
        self.print_fn(f"[AblationSummary] {summary}")

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f"[!] data loader keys: {self.loader_dict.keys()}")

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, epoch, best_eval_acc, logger=None):
        self.model.train()
        if self.use_ema_teacher:
            self.ema = EMA(self.model, self.ema_m)
            self.ema.register()
            if args.resume and self.ema_model is not None:
                self.ema.load(self.ema_model)
        else:
            self.ema = None

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        warm_meter = AverageMeter()
        sup_meter = AverageMeter()
        unsup_meter = AverageMeter()
        orth_meter = AverageMeter()
        align_meter = AverageMeter()
        total_meter = AverageMeter()
        clean_ratio_meter = AverageMeter()
        lr_last = 0.0
        epoch_start_time = time.perf_counter()
        if torch.cuda.is_available() and (args.gpu is not None):
            torch.cuda.reset_peak_memory_stats(args.gpu)

        data_iter = zip(self.loader_dict["train_lb"], self.loader_dict["train_ulb"])
        module3_on = bool(getattr(self.model, "use_cd_state_modeling", False))
        for (_, x_lb, t_lb, y_lb), (_, x_ulb_w, x_ulb_s0, x_ulb_s1, t_ulb, y_ulb) in tqdm(
            data_iter, total=len(self.loader_dict["train_ulb"])
        ):
            del x_ulb_s0, x_ulb_s1
            x_lb = x_lb.cuda(args.gpu)
            x_ulb_w = x_ulb_w.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            y_ulb = y_ulb.cuda(args.gpu)

            text_lb = self._to_text_list(t_lb)
            # ===== Module-4 OFF branch =====
            # Pure supervised CE only, all robust losses removed.
            if not self.noise_robust_on:
                text_inputs_lb = self._tokenize(text_lb, device=args.gpu)
                self.optimizer.zero_grad(set_to_none=True)
                with amp_cm():
                    output_lb = self.model(x_lb, text_inputs_lb)
                    # sup_loss source:
                    # - output_lb["pre_f"] (final classifier logits)
                    no_noise_supervision = str(getattr(args, "cdr_no_noise_supervision", "ce")).lower()
                    if no_noise_supervision == "ce_static_smoothing":
                        eps = float(getattr(args, "cdr_static_smoothing_eps", 0.1))
                        eps = min(max(eps, 0.0), 1.0)
                        y_smooth = F.one_hot(y_lb, num_classes=self.num_classes).float()
                        y_smooth = (1.0 - eps) * y_smooth + eps / float(self.num_classes)
                        sup_loss = soft_ce_loss(output_lb["pre_f"], y_smooth, reduction="mean")
                    else:
                        sup_loss = F.cross_entropy(output_lb["pre_f"], y_lb, reduction="mean")
                    warm_loss = sup_loss
                    unsup_loss = torch.zeros(1, device=x_lb.device).squeeze(0)
                    orth_loss = torch.zeros(1, device=x_lb.device).squeeze(0)
                    align_loss = torch.zeros(1, device=x_lb.device).squeeze(0)
                    total_loss = sup_loss
                    clean_ratio = torch.ones(1, device=x_lb.device).squeeze(0)

                if args.amp:
                    scaler.scale(total_loss).backward()
                    if args.clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if args.clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                lr_last = self.optimizer.param_groups[0]["lr"]

                warm_meter.update(warm_loss.detach().cpu().item())
                sup_meter.update(sup_loss.detach().cpu().item())
                unsup_meter.update(unsup_loss.detach().cpu().item())
                orth_meter.update(orth_loss.detach().cpu().item())
                align_meter.update(align_loss.detach().cpu().item())
                total_meter.update(total_loss.detach().cpu().item())
                clean_ratio_meter.update(clean_ratio.detach().cpu().item())
                continue

            text_ulb = self._to_text_list(t_ulb)

            images = torch.cat([x_lb, x_ulb_w], dim=0)
            labels = torch.cat([y_lb, y_ulb], dim=0)
            text_all = text_lb + text_ulb
            text_inputs = self._tokenize(text_all, device=args.gpu)

            self.optimizer.zero_grad(set_to_none=True)
            # ===== Module-4 ON branch =====
            # Full noise-robust training pipeline.
            with amp_cm():
                output = self.model(images, text_inputs)
                logits_f = output["pre_f"]
                logits_c = output["pre_c"]
                logits_t = output["pre_t"]
                logits_v = output["pre_v"]
                p_f = torch.softmax(logits_f, dim=-1)
                has_pre_c = module3_on and (logits_c is not None)
                p_c = torch.softmax(logits_c, dim=-1) if has_pre_c else None
                p_t = torch.softmax(logits_t, dim=-1)
                p_v = torch.softmax(logits_v, dim=-1)
                ema_output = self._forward_with_ema(images, text_inputs)
                p_ema = torch.softmax(ema_output["pre_f"], dim=-1)

                if epoch < args.cdr_warmup_epochs:
                    # warmup supervised robust loss block
                    warm_loss = gce_loss(logits_f, labels, q=args.cdr_gce_q, reduction="mean")
                    if has_pre_c:
                        warm_loss = warm_loss + args.cdr_lambda1 * gce_loss(
                            logits_c, labels, q=args.cdr_gce_q, reduction="mean"
                        )
                    warm_loss = warm_loss + args.cdr_lambda2 * (
                        gce_loss(logits_t, labels, q=args.cdr_gce_q, reduction="mean")
                        + gce_loss(logits_v, labels, q=args.cdr_gce_q, reduction="mean")
                    )
                    sup_loss = warm_loss
                    unsup_loss = torch.zeros(1, device=images.device).squeeze(0)
                    orth_loss = torch.zeros(1, device=images.device).squeeze(0)
                    align_loss = torch.zeros(1, device=images.device).squeeze(0)
                    total_loss = warm_loss
                    clean_ratio = torch.ones(1, device=images.device).squeeze(0)
                else:
                    # robust relabel + clean/noisy split block
                    n_all = labels.size(0)
                    n_lb = y_lb.size(0)
                    n_ulb = max(0, n_all - n_lb)

                    onehot = F.one_hot(labels, num_classes=self.num_classes).float()
                    if has_pre_c:
                        q = self._sharpen((p_f + p_c + p_ema) / 3.0, args.cdr_sharpen_temp)
                    else:
                        # strict drop path: remove pre_c from soft relabel.
                        q = self._sharpen((p_f + p_ema) / 2.0, args.cdr_sharpen_temp)

                    # Labeled samples are always treated as clean with hard labels.
                    y_bar = onehot.clone()
                    clean_mask = torch.zeros(n_all, dtype=torch.bool, device=images.device)
                    noisy_mask = torch.zeros(n_all, dtype=torch.bool, device=images.device)
                    clean_mask[:n_lb] = True

                    # Clean/noisy estimation is only applied to unlabeled subset.
                    if n_ulb > 0:
                        sl = slice(n_lb, n_all)
                        l_f_ulb = gce_loss(logits_f[sl], labels[sl], q=args.cdr_gce_q, reduction="none")
                        l_t_ulb = gce_loss(logits_t[sl], labels[sl], q=args.cdr_gce_q, reduction="none")
                        l_v_ulb = gce_loss(logits_v[sl], labels[sl], q=args.cdr_gce_q, reduction="none")
                        u_ulb = 1.0 - js_divergence_prob(p_t[sl], p_v[sl])
                        score_ulb = u_ulb * l_f_ulb + (1.0 - u_ulb) * torch.minimum(l_t_ulb, l_v_ulb)
                        score_ulb = score_ulb + args.cdr_beta * kl_divergence_prob(p_f[sl], p_ema[sl])
                        clean_prob_ulb = self._estimate_clean_prob(score_ulb)

                        y_bar[sl] = (
                            clean_prob_ulb.unsqueeze(1) * onehot[sl]
                            + (1.0 - clean_prob_ulb.unsqueeze(1)) * q[sl]
                        )
                        clean_mask[sl] = clean_prob_ulb > args.cdr_clean_threshold
                        noisy_mask[sl] = ~clean_mask[sl]
                        clean_ratio = clean_mask[sl].float().mean()
                    else:
                        clean_ratio = torch.ones(1, device=images.device).squeeze(0)

                    if clean_mask.any():
                        # sup_loss source in robust stage:
                        # - pre_f always contributes
                        # - pre_c contributes only when Module-3 CD is enabled
                        # - pre_t/pre_v contributes as auxiliary supervision
                        sup_loss = soft_ce_loss(logits_f[clean_mask], y_bar[clean_mask], reduction="mean")
                        if has_pre_c:
                            sup_loss = sup_loss + args.cdr_lambda1 * soft_ce_loss(
                                logits_c[clean_mask], y_bar[clean_mask], reduction="mean"
                            )
                        sup_loss = sup_loss + args.cdr_lambda2 * (
                            soft_ce_loss(logits_t[clean_mask], y_bar[clean_mask], reduction="mean")
                            + soft_ce_loss(logits_v[clean_mask], y_bar[clean_mask], reduction="mean")
                        )
                    else:
                        sup_loss = torch.zeros(1, device=images.device).squeeze(0)

                    if noisy_mask.any():
                        # unsup consistency on noisy subset
                        noisy_indices = noisy_mask.nonzero(as_tuple=True)[0].tolist()
                        noisy_images = images[noisy_mask]
                        noisy_text = [text_all[idx] for idx in noisy_indices]
                        noisy_text_1 = self._tokenize(noisy_text, device=args.gpu)
                        noisy_text_2 = self._tokenize(noisy_text, device=args.gpu)
                        out_1 = self.model(noisy_images, noisy_text_1)
                        out_2 = self.model(noisy_images, noisy_text_2)
                        p_f_1 = torch.softmax(out_1["pre_f"], dim=-1)
                        p_f_2 = torch.softmax(out_2["pre_f"], dim=-1)
                        unsup_loss = js_divergence_prob(p_f_1, p_f_2).mean()
                        if has_pre_c and (out_1["pre_c"] is not None) and (out_2["pre_c"] is not None):
                            p_c_1 = torch.softmax(out_1["pre_c"], dim=-1)
                            p_c_2 = torch.softmax(out_2["pre_c"], dim=-1)
                            unsup_loss = unsup_loss + args.cdr_lambda3 * js_divergence_prob(p_c_1, p_c_2).mean()
                    else:
                        unsup_loss = torch.zeros(1, device=images.device).squeeze(0)

                    can_use_reg = has_pre_c and (output["h_c"] is not None) and (output["rho"] is not None)
                    if can_use_reg and self.use_orth_loss:
                        orth_loss = self._orth_loss(output["h_c"], output["h_dt"], output["h_dv"])
                    else:
                        orth_loss = torch.zeros(1, device=images.device).squeeze(0)

                    if can_use_reg and self.use_align_loss:
                        align_loss = self._align_loss(p_t, p_v, output["rho"], args)
                    else:
                        align_loss = torch.zeros(1, device=images.device).squeeze(0)
                    total_loss = (
                        sup_loss
                        + args.cdr_lambda4 * unsup_loss
                        + args.cdr_lambda5 * orth_loss
                        + args.cdr_lambda6 * align_loss
                    )
                    warm_loss = torch.zeros(1, device=images.device).squeeze(0)

            if args.amp:
                scaler.scale(total_loss).backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update()
            lr_last = self.optimizer.param_groups[0]["lr"]

            warm_meter.update(warm_loss.detach().cpu().item())
            sup_meter.update(sup_loss.detach().cpu().item())
            unsup_meter.update(unsup_loss.detach().cpu().item())
            orth_meter.update(orth_loss.detach().cpu().item())
            align_meter.update(align_loss.detach().cpu().item())
            total_meter.update(total_loss.detach().cpu().item())
            clean_ratio_meter.update(clean_ratio.detach().cpu().item())

        self.print_fn(
            "Epoch {}/{} train: lr={}, warm_loss={}, sup_loss={}, unsup_loss={}, "
            "orth_loss={}, align_loss={}, total_loss={}, clean_ratio={}".format(
                epoch,
                args.epoch,
                lr_last,
                warm_meter.avg,
                sup_meter.avg,
                unsup_meter.avg,
                orth_meter.avg,
                align_meter.avg,
                total_meter.avg,
                clean_ratio_meter.avg,
            )
        )
        train_seconds = float(time.perf_counter() - epoch_start_time)
        max_gpu_mem_mb = 0.0
        if torch.cuda.is_available() and (args.gpu is not None):
            max_gpu_mem_mb = float(torch.cuda.max_memory_allocated(args.gpu) / (1024.0 * 1024.0))
        self.print_fn(
            "Epoch {}/{} runtime: train_seconds={}, max_gpu_mem_mb={}".format(
                epoch,
                args.epoch,
                train_seconds,
                max_gpu_mem_mb,
            )
        )

        last_n = max(0, int(args.cdr_mispred_last_n_epochs))
        start_save_epoch = max(0, int(args.epoch) - last_n)
        should_save_mispred = bool(args.cdr_save_mispred and epoch >= start_save_epoch)
        if args.cdr_save_mispred and epoch == start_save_epoch:
            self.print_fn(
                f"Misprediction export starts at epoch {start_save_epoch} "
                f"(last {last_n} epochs of total {args.epoch})."
            )
        eval_dict = self.evaluate(args=args, epoch=epoch, save_mispred=should_save_mispred)
        eval_dict["train/epoch-seconds"] = train_seconds
        eval_dict["train/max-gpu-memory-mb"] = max_gpu_mem_mb
        best_eval_acc = max(best_eval_acc, eval_dict["eval/top-1-acc"])
        self.print_fn(
            "Epoch {}/{} test: test loss: {}, top-1 acc: {}, top-{} acc: {}, best top-1 acc: {}".format(
                epoch,
                args.epoch,
                eval_dict["eval/loss"],
                eval_dict["eval/top-1-acc"],
                int(eval_dict["eval/top-k"]),
                eval_dict["eval/top-k-acc"],
                best_eval_acc,
            )
        )
        self.print_fn(
            "Epoch {}/{} pred dist: count={}, ratio={}".format(
                epoch,
                args.epoch,
                eval_dict["eval/pred-dist-count"],
                eval_dict["eval/pred-dist-ratio"],
            )
        )
        self.print_fn(
            "Epoch {}/{} true dist: count={}, ratio={}".format(
                epoch,
                args.epoch,
                eval_dict["eval/true-dist-count"],
                eval_dict["eval/true-dist-ratio"],
            )
        )
        self.print_fn(
            "Epoch {}/{} seq-bucket acc: short={}({}), medium={}({}), long={}({}), "
            "len_bounds: short<= {}, medium<= {}, long>= {}".format(
                epoch,
                args.epoch,
                eval_dict["eval/seq-short-top1-acc"],
                eval_dict["eval/seq-short-count"],
                eval_dict["eval/seq-medium-top1-acc"],
                eval_dict["eval/seq-medium-count"],
                eval_dict["eval/seq-long-top1-acc"],
                eval_dict["eval/seq-long-count"],
                eval_dict["eval/seq-short-max-len"],
                eval_dict["eval/seq-medium-max-len"],
                eval_dict["eval/seq-long-min-len"],
            )
        )
        self.print_fn(
            "Epoch {}/{} high-conflict(top {}): acc={}, count={}, threshold={}".format(
                epoch,
                args.epoch,
                eval_dict["eval/high-conflict-ratio"],
                eval_dict["eval/high-conflict-top1-acc"],
                eval_dict["eval/high-conflict-count"],
                eval_dict["eval/high-conflict-threshold"],
            )
        )
        self._append_epoch_history(
            args=args,
            epoch=epoch,
            train_stats={
                "lr": float(lr_last),
                "warm_loss": float(warm_meter.avg),
                "sup_loss": float(sup_meter.avg),
                "unsup_loss": float(unsup_meter.avg),
                "orth_loss": float(orth_meter.avg),
                "align_loss": float(align_meter.avg),
                "total_loss": float(total_meter.avg),
                "clean_ratio": float(clean_ratio_meter.avg),
                "epoch_seconds": train_seconds,
                "max_gpu_memory_mb": max_gpu_mem_mb,
            },
            eval_dict=eval_dict,
        )

        save_path = os.path.join(args.save_dir, args.save_name)
        if eval_dict["eval/top-1-acc"] == best_eval_acc:
            self.save_model("model_best.pth", save_path)

        return eval_dict["eval/top-1-acc"]

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, epoch=0, save_mispred=False):
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict["eval"]

        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        seq_lens = []
        deltas = []
        mispred_saved = 0
        for idx, x, text_input, y in eval_loader:
            x = x.cuda(args.gpu)
            y = y.cuda(args.gpu)
            text_list = self._to_text_list(text_input)
            text_tokens = self._tokenize(text_list, device=args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch

            output = self.model(x, text_tokens)
            logits = output["pre_f"]
            loss = F.cross_entropy(logits, y, reduction="mean")

            y_true.extend(y.cpu().tolist())
            pred = torch.max(logits, dim=-1)[1]
            y_pred.extend(pred.cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            if "attention_mask" in text_tokens:
                seq_lens.extend(text_tokens["attention_mask"].sum(dim=1).detach().cpu().tolist())
            else:
                seq_lens.extend([int(text_tokens["input_ids"].shape[1])] * num_batch)
            if output.get("delta", None) is not None:
                deltas.extend(output["delta"].detach().float().cpu().tolist())
            else:
                deltas.extend([0.0] * num_batch)
            total_loss += loss.cpu().item() * num_batch
            if save_mispred:
                mispred_saved += self._save_mispred_batch(
                    args=args,
                    epoch=epoch,
                    sample_idx=idx,
                    x=x.detach(),
                    text_list=text_list,
                    y_true=y.detach(),
                    y_pred=pred.detach(),
                    already_saved=mispred_saved,
                )

        top1 = accuracy_score(y_true, y_pred)
        k = min(5, max(1, self.num_classes - 1))
        topk_acc = top_k_accuracy_score(y_true, y_logits, k=k, labels=list(range(self.num_classes)))
        pred_count = np.bincount(np.array(y_pred, dtype=np.int64), minlength=self.num_classes)
        pred_ratio = pred_count / max(int(pred_count.sum()), 1)
        true_count = np.bincount(np.array(y_true, dtype=np.int64), minlength=self.num_classes)
        true_ratio = true_count / max(int(true_count.sum()), 1)
        seq_bucket_metrics = self._compute_seq_bucket_metrics(seq_lens, y_true, y_pred, args)
        high_conflict_metrics = self._compute_high_conflict_metrics(deltas, y_true, y_pred, args)

        if self.ema is not None:
            self.ema.restore()
        self.model.train()
        if save_mispred and mispred_saved > 0:
            self.print_fn(
                f"Epoch {epoch}: saved {mispred_saved} misclassified samples to "
                f"{os.path.join(args.save_dir, args.save_name, args.cdr_mispred_subdir, f'epoch_{epoch:04d}')}"
            )
        return {
            "eval/loss": total_loss / max(total_num, 1.0),
            "eval/top-1-acc": top1,
            "eval/top-k": k,
            "eval/top-k-acc": topk_acc,
            "eval/pred-dist-count": pred_count.tolist(),
            "eval/pred-dist-ratio": [float(x) for x in pred_ratio.tolist()],
            "eval/true-dist-count": true_count.tolist(),
            "eval/true-dist-ratio": [float(x) for x in true_ratio.tolist()],
            **seq_bucket_metrics,
            **high_conflict_metrics,
        }

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
            ema_model = self.model.state_dict()
            self.ema.restore()
        else:
            ema_model = None
        self.model.train()

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "it": self.it,
                "ema_model": ema_model,
            },
            save_filename,
        )
        if self.num_classes == 10:
            tb_path = os.path.join(save_path, "tensorboard")
            if not os.path.exists(tb_path):
                os.makedirs(tb_path, exist_ok=True)
            with open(os.path.join(save_path, "tensorboard", "lst_fix.pkl"), "wb") as f:
                pickle.dump([], f)
            with open(os.path.join(save_path, "tensorboard", "abs_lst.pkl"), "wb") as h:
                pickle.dump([], h)
            with open(os.path.join(save_path, "tensorboard", "clsacc.pkl"), "wb") as g:
                pickle.dump([], g)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint["model"])
        if checkpoint.get("ema_model", None) is not None:
            self.ema_model = deepcopy(self.model)
            self.ema_model.load_state_dict(checkpoint["ema_model"])
        else:
            self.ema_model = None
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.it = checkpoint["it"]
        self.print_fn("model loaded")

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]: offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
