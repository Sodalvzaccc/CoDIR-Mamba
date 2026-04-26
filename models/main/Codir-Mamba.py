"""
Helper utilities extracted from the CDR training pipeline.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torchvision.utils import save_image


def gce_loss(logits, targets, q=0.7, reduction="mean"):
    probs = torch.softmax(logits, dim=-1)
    py = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
    loss = (1.0 - py.pow(q)) / q
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def soft_ce_loss(logits, soft_targets, reduction="mean"):
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(soft_targets * log_prob, dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def js_divergence_prob(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence_prob(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


class CDRMethodHelperMixin:
    def _to_text_list(self, text_batch):
        text_list = []
        for item in text_batch:
            if isinstance(item, str):
                text_list.append(item)
            elif hasattr(item, "item"):
                text_list.append(str(item.item()))
            else:
                text_list.append(str(item))
        return text_list

    def _tokenize(self, text_list, device):
        text_input = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.cuda(device) for k, v in text_input.items()}

    def _estimate_clean_prob(self, score):
        # DivideMix-style two-component GMM on score distribution.
        if score.numel() < 2:
            return torch.ones_like(score) * 0.5
        score_np = score.detach().float().cpu().view(-1, 1).numpy()
        try:
            gmm = GaussianMixture(
                n_components=2, covariance_type="full", max_iter=100, reg_covar=1e-4, random_state=0
            )
            gmm.fit(score_np)
            comp_probs = gmm.predict_proba(score_np)
            clean_comp = np.argmin(gmm.means_.reshape(-1))
            clean_prob = comp_probs[:, clean_comp]
            return torch.from_numpy(clean_prob).to(score.device, dtype=score.dtype)
        except Exception:
            s_min = score.min()
            s_max = score.max()
            norm = (score - s_min) / (s_max - s_min + 1e-8)
            return 1.0 - norm

    def _sharpen(self, probs, temp):
        temp = max(temp, 1e-6)
        p = probs.pow(1.0 / temp)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _orth_loss(self, h_c, h_dt, h_dv):
        # Strategy-A safe orthogonality:
        # hc, hdt, hdv: [B, D]
        hc = F.normalize(h_c, dim=-1)
        hdt = F.normalize(h_dt, dim=-1)
        hdv = F.normalize(h_dv, dim=-1)
        loss_ct = ((hc * hdt).sum(dim=-1).pow(2)).mean()
        loss_cv = ((hc * hdv).sum(dim=-1).pow(2)).mean()
        return 0.5 * (loss_ct + loss_cv)

    def _align_loss(self, p_t, p_v, rho, args):
        # Use temperature on rho to avoid exp underflow.
        tau = max(args.cdr_align_tau, 1e-6)
        weight = torch.exp(-rho / tau)
        return (weight * js_divergence_prob(p_t, p_v)).mean()

    def _get_norm_mean_std(self, dataset_name):
        key = str(dataset_name).lower()
        if key in self.image_norm_stats:
            return self.image_norm_stats[key]
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def _save_mispred_batch(self, args, epoch, sample_idx, x, text_list, y_true, y_pred, already_saved=0):
        if not args.cdr_save_mispred:
            return 0
        wrong = (y_true != y_pred).nonzero(as_tuple=True)[0]
        if wrong.numel() == 0:
            return 0

        save_root = os.path.join(
            args.save_dir, args.save_name, args.cdr_mispred_subdir, f"epoch_{epoch:04d}"
        )
        os.makedirs(save_root, exist_ok=True)
        meta_path = os.path.join(save_root, "meta.jsonl")

        mean, std = self._get_norm_mean_std(args.dataset)
        mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)

        num_saved = 0
        max_to_save = max(0, args.cdr_mispred_max_per_epoch)
        for wid in wrong.tolist():
            if max_to_save > 0 and (already_saved + num_saved) >= max_to_save:
                break
            img_idx = int(sample_idx[wid]) if hasattr(sample_idx[wid], "__int__") else wid
            pred = int(y_pred[wid].item())
            true = int(y_true[wid].item())
            img = (x[wid: wid + 1] * std_t + mean_t).clamp(0.0, 1.0)
            filename = f"idx_{img_idx}_true_{true}_pred_{pred}.png"
            img_path = os.path.join(save_root, filename)
            save_image(img, img_path)

            record = {
                "epoch": int(epoch),
                "sample_idx": img_idx,
                "true_label": true,
                "pred_label": pred,
                "image_path": filename,
                "text": str(text_list[wid]),
            }
            with open(meta_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_saved += 1
        return num_saved

    def _compute_seq_bucket_metrics(self, seq_lens, y_true, y_pred, args):
        if len(seq_lens) == 0:
            return {
                "eval/seq-short-count": 0,
                "eval/seq-short-top1-acc": 0.0,
                "eval/seq-medium-count": 0,
                "eval/seq-medium-top1-acc": 0.0,
                "eval/seq-long-count": 0,
                "eval/seq-long-top1-acc": 0.0,
                "eval/seq-short-max-len": 0,
                "eval/seq-medium-max-len": 0,
                "eval/seq-long-min-len": 0,
            }

        q1 = float(getattr(args, "cdr_seq_bucket_q1", 0.33))
        q2 = float(getattr(args, "cdr_seq_bucket_q2", 0.67))
        q1 = min(max(q1, 0.0), 1.0)
        q2 = min(max(q2, 0.0), 1.0)
        if q2 < q1:
            q1, q2 = q2, q1

        lens = np.asarray(seq_lens, dtype=np.float32)
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        correct = (y_true_arr == y_pred_arr).astype(np.float32)

        b1 = float(np.quantile(lens, q1))
        b2 = float(np.quantile(lens, q2))
        short_mask = lens <= b1
        medium_mask = (lens > b1) & (lens <= b2)
        long_mask = lens > b2

        def _safe_acc(mask):
            n = int(mask.sum())
            if n <= 0:
                return 0, 0.0
            return n, float(correct[mask].mean())

        n_short, a_short = _safe_acc(short_mask)
        n_mid, a_mid = _safe_acc(medium_mask)
        n_long, a_long = _safe_acc(long_mask)

        return {
            "eval/seq-short-count": n_short,
            "eval/seq-short-top1-acc": a_short,
            "eval/seq-medium-count": n_mid,
            "eval/seq-medium-top1-acc": a_mid,
            "eval/seq-long-count": n_long,
            "eval/seq-long-top1-acc": a_long,
            "eval/seq-short-max-len": int(round(b1)),
            "eval/seq-medium-max-len": int(round(b2)),
            "eval/seq-long-min-len": int(round(b2)) + 1,
        }

    def _compute_high_conflict_metrics(self, deltas, y_true, y_pred, args):
        if len(deltas) == 0:
            return {
                "eval/high-conflict-ratio": 0.0,
                "eval/high-conflict-threshold": 0.0,
                "eval/high-conflict-count": 0,
                "eval/high-conflict-top1-acc": 0.0,
            }

        ratio = float(getattr(args, "cdr_high_conflict_ratio", 0.3))
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio <= 0:
            return {
                "eval/high-conflict-ratio": ratio,
                "eval/high-conflict-threshold": 0.0,
                "eval/high-conflict-count": 0,
                "eval/high-conflict-top1-acc": 0.0,
            }

        delta_arr = np.asarray(deltas, dtype=np.float32)
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        correct = (y_true_arr == y_pred_arr).astype(np.float32)

        q = 1.0 - ratio
        threshold = float(np.quantile(delta_arr, q))
        mask = delta_arr >= threshold
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count > 0 else 0.0
        return {
            "eval/high-conflict-ratio": ratio,
            "eval/high-conflict-threshold": threshold,
            "eval/high-conflict-count": count,
            "eval/high-conflict-top1-acc": acc,
        }

    def _append_epoch_history(self, args, epoch, train_stats, eval_dict):
        save_path = os.path.join(args.save_dir, args.save_name)
        os.makedirs(save_path, exist_ok=True)
        history_path = os.path.join(save_path, "metrics_history.jsonl")
        payload = {
            "epoch": int(epoch),
            "train": train_stats,
            "eval": eval_dict,
        }
        with open(history_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _forward_with_ema(self, images, text_inputs):
        if not self.use_ema_teacher:
            with torch.no_grad():
                return self.model(images, text_inputs)
        with torch.no_grad():
            was_training = self.model.training
            self.model.eval()
            self.ema.apply_shadow()
            ema_out = self.model(images, text_inputs)
            self.ema.restore()
            if was_training:
                self.model.train()
            return ema_out
