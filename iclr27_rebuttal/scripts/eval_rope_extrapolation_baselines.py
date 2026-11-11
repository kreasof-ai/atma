#!/usr/bin/env python3
"""Evaluate Training-Free RoPE Extrapolation Baselines on the 9.816B RoPE Checkpoint.

Directly addresses Reviewer ihsT (Weakness 1 & Question 1) and Reviewer HE3D (Question 3):
Evaluates the matched 9.816B-token RoPE + gated-delta memory checkpoint under exact
checkpoint-faithful half-truncated RoPE (`base = 1024.0`, `dim // 4` active frequencies +
`dim // 4` zero-frequency channels, matching `train.model.Rotary`):

1. `vanilla`: Unmodified checkpoint RoPE (`L_train = 2048`, `base = 1024.0`).
2. `pi`: Linear Position Interpolation (`inv_freq / scale` for `L > L_train`).
3. `dynamic_ntk`: Context-dependent NTK-aware base scaling on the rotated subspace
   `base' = base * scale^(d_rot / (d_rot - 2))` with `d_rot = dim // 2`.
4. `yarn`: NTK-by-parts frequency ramp interpolation (`beta_fast = 32`, `beta_slow = 1`)
   on the active RoPE subspace + YaRN attention logit temperature scaling
   `mscale = 0.1 * ln(scale) + 1`.
5. `rerope`: Rectified Rotary Position Embeddings (Su, 2023), clamping relative RoPE
   distances at `min(i - j, w)` with `w = L_train // 2 = 1024` via exact tiled online
   softmax (plus `rerope_log` adding `max(1, ln(i+1)/ln(L_train))` query scaling).

Includes an automatic startup equivalence assertion verifying `max_abs_diff == 0.0`
between `mode="vanilla"` and unmodified `train.model.Rotary` before running benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scoring import DirectScorer
from supplementary.pat_feedback.run_e2_experiment import evaluate_bpb, evaluate_retrieval, load_fixtures

ROTARY_SPEC_VERSION = "half_truncated_base1024_v2"
BASE_THETA = 1024.0
TRAIN_LEN = 2048


def resolve_rope_snapshot() -> str:
    candidates = [
        (
            "/home/sagemaker-user/.cache/huggingface/hub/"
            "models--anon--atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0/"
            "snapshots/de0a2f1c63b0631a040a2d6adac9e2d43fd5c581"
        ),
        (
            "/home/sagemaker-user/.cache/huggingface/hub/"
            "models--ChavyvAkvar--atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0/"
            "snapshots/de0a2f1c63b0631a040a2d6adac9e2d43fd5c581"
        ),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[0]


class ExtendedRotary(torch.nn.Module):
    """Exact drop-in replacement for `train.model.Rotary` (half-truncated, base=1024)
    supporting Vanilla, Position Interpolation (PI), Dynamic-NTK, and YaRN."""

    def __init__(
        self,
        dim: int,
        base: float = BASE_THETA,
        train_len: int = TRAIN_LEN,
        mode: str = "vanilla",
        ntk_alpha: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.rot_pairs = self.dim // 4  # 32 rotated pairs (64 dims) + 32 unrotated pairs (64 dims)
        self.rot_dim = self.dim // 2    # 64 active rotated dimensions
        self.base = float(base)
        self.train_len = int(train_len)
        self.mode = mode
        self.ntk_alpha = float(ntk_alpha)
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)

        active_freq = (1.0 / self.base) ** torch.linspace(0.0, 1.0, steps=self.rot_pairs, dtype=torch.float32)
        angular_freq = torch.cat([active_freq, active_freq.new_zeros(self.rot_pairs)])
        self.register_buffer("angular_freq", angular_freq, persistent=False)

    def _pad_half_truncated(self, active_freq: Tensor) -> Tensor:
        return torch.cat([active_freq, active_freq.new_zeros(self.rot_pairs)])

    def _yarn_angular_freq_and_mscale(self, seq_len: int, device: torch.device) -> tuple[Tensor, float]:
        if seq_len <= self.train_len:
            return self.angular_freq.to(device), 1.0

        scale = float(seq_len) / float(self.train_len)
        u = torch.linspace(0.0, 1.0, steps=self.rot_pairs, dtype=torch.float32, device=device)
        freq_extrap = (1.0 / self.base) ** u
        freq_interp = freq_extrap / scale

        # Wavelength at normalized index u in [0, 1] is 2 * pi * (base ** u).
        # Number of rotations in train_len is r(u) = train_len / (2 * pi * base ** u).
        # Solving r(u) = beta gives u = ln(train_len / (2 * pi * beta)) / ln(base).
        u_low = math.log(self.train_len / (self.beta_fast * 2.0 * math.pi)) / math.log(self.base)
        u_high = math.log(self.train_len / (self.beta_slow * 2.0 * math.pi)) / math.log(self.base)
        idx_low = max(math.floor(u_low * (self.rot_pairs - 1)), 0)
        idx_high = min(math.ceil(u_high * (self.rot_pairs - 1)), self.rot_pairs - 1)
        if idx_low >= idx_high:
            idx_high = idx_low + 1e-3

        idx = torch.arange(self.rot_pairs, dtype=torch.float32, device=device)
        ramp = torch.clamp((idx - idx_low) / (idx_high - idx_low), 0.0, 1.0)
        # ramp == 0 (high-freq local channels) -> extrapolate; ramp == 1 (low-freq global) -> interpolate
        active_freq = freq_interp * ramp + freq_extrap * (1.0 - ramp)
        mscale = 0.1 * math.log(scale) + 1.0
        return self._pad_half_truncated(active_freq), mscale

    def rotate_fixed_position(self, x_BTHD: Tensor, fixed_pos: float) -> Tensor:
        """Rotate all tokens by a constant position `fixed_pos` (used by ReRoPE)."""
        af = self.angular_freq.to(x_BTHD.device)
        theta = (float(fixed_pos) * af)[None, None, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

    def forward(self, x_BTHD: Tensor) -> Tensor:
        seq_len = x_BTHD.size(1)
        device = x_BTHD.device
        mscale = 1.0

        if self.mode == "vanilla" or seq_len <= self.train_len:
            angular_freq = self.angular_freq.to(device)
        elif self.mode == "pi":
            scale = float(seq_len) / float(self.train_len)
            angular_freq = self.angular_freq.to(device) / scale
        elif self.mode == "dynamic_ntk":
            scale = (self.ntk_alpha * seq_len / self.train_len) - (self.ntk_alpha - 1.0)
            scaled_base = self.base * (scale ** (self.rot_dim / (self.rot_dim - 2.0)))
            u = torch.linspace(0.0, 1.0, steps=self.rot_pairs, dtype=torch.float32, device=device)
            active_freq = (1.0 / scaled_base) ** u
            angular_freq = self._pad_half_truncated(active_freq)
        elif self.mode == "yarn":
            angular_freq, mscale = self._yarn_angular_freq_and_mscale(seq_len, device)
        else:
            raise ValueError(f"Unknown rotary mode: {self.mode}")

        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(pos, angular_freq)[None, :, None, :]
        cos = theta.cos() * mscale
        sin = theta.sin() * mscale
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


def _rerope_tiled_sdpa(
    q_raw: Tensor,
    k_raw: Tensor,
    v_raw: Tensor,
    rotary: ExtendedRotary,
    groups: int,
    scale: float,
    window: int = 1024,
    log_scale: bool = False,
    train_len: int = TRAIN_LEN,
    q_tile: int = 256,
    k_tile: int = 16384,
) -> Tensor:
    """Exact tiled online-softmax ReRoPE (Su, 2023) with O(q_tile * k_tile) scratch memory.

    For causal pair j <= i, relative position is clamped at min(i - j, window):
      - If i - j < window: <R_i q_i, R_j k_j>
      - If i - j >= window: <R_window q_i, k_j>
    """
    B, T, H, dk = q_raw.shape
    device = q_raw.device
    dtype = q_raw.dtype

    q_std = rotary(q_raw)
    k_std = rotary(k_raw).repeat_interleave(groups, dim=2)
    q_rect = rotary.rotate_fixed_position(q_raw, float(window))
    k_zero = k_raw.repeat_interleave(groups, dim=2)
    v_exp = v_raw.repeat_interleave(groups, dim=2)

    if log_scale:
        pos_1 = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        lscale = torch.clamp(torch.log(pos_1) / math.log(float(train_len)), min=1.0).view(1, T, 1, 1)
        q_std = (q_std.float() * lscale).to(dtype)
        q_rect = (q_rect.float() * lscale).to(dtype)

    # Fast exact SDPA path when the entire sequence fits inside `window`
    if T <= window:
        return F.scaled_dot_product_attention(
            q_std.transpose(1, 2),
            k_std.transpose(1, 2),
            v_exp.transpose(1, 2),
            is_causal=True,
            scale=scale,
        ).transpose(1, 2)

    q_std_h = q_std.transpose(1, 2).contiguous()          # (B, H, T, dk)
    q_rect_h = q_rect.transpose(1, 2).contiguous()        # (B, H, T, dk)
    k_std_ht = k_std.permute(0, 2, 3, 1).contiguous()     # (B, H, dk, T)
    k_zero_ht = k_zero.permute(0, 2, 3, 1).contiguous()   # (B, H, dk, T)
    v_exp_h = v_exp.transpose(1, 2).contiguous()          # (B, H, T, dk)

    out_h = torch.empty((B, H, T, dk), device=device, dtype=dtype)

    for q0 in range(0, T, q_tile):
        q1 = min(T, q0 + q_tile)
        k_split = max(0, q0 - window)

        qi = torch.arange(q0, q1, device=device).view(-1, 1)
        kj = torch.arange(k_split, q1, device=device).view(1, -1)
        dist = qi - kj

        std_near = torch.matmul(q_std_h[:, :, q0:q1], k_std_ht[:, :, :, k_split:q1]).float() * scale
        if k_split == 0 and q1 <= window:
            near_logits = std_near
        else:
            rect_near = torch.matmul(q_rect_h[:, :, q0:q1], k_zero_ht[:, :, :, k_split:q1]).float() * scale
            near_logits = torch.where((dist < window)[None, None, :, :], std_near, rect_near)
        near_logits = near_logits.masked_fill((dist < 0)[None, None, :, :], float("-inf"))

        m_acc = near_logits.amax(dim=-1, keepdim=True)
        p_near = torch.exp(near_logits - m_acc)
        l_acc = p_near.sum(dim=-1, keepdim=True)
        o_acc = torch.matmul(p_near.to(dtype), v_exp_h[:, :, k_split:q1, :]).float()

        qr_slice = q_rect_h[:, :, q0:q1]
        for ks0 in range(0, k_split, k_tile):
            ks1 = min(k_split, ks0 + k_tile)
            far_logits = torch.matmul(qr_slice, k_zero_ht[:, :, :, ks0:ks1]).float() * scale
            m_far = far_logits.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m_acc, m_far)
            alpha_old = torch.exp(m_acc - m_new)
            p_far = torch.exp(far_logits - m_new)
            o_acc = o_acc * alpha_old + torch.matmul(p_far.to(dtype), v_exp_h[:, :, ks0:ks1, :]).float()
            l_acc = l_acc * alpha_old + p_far.sum(dim=-1, keepdim=True)
            m_acc = m_new

        out_h[:, :, q0:q1, :] = (o_acc / l_acc).to(dtype)

    return out_h.transpose(1, 2)


def _make_rerope_attn_forward(window: int, log_scale: bool):
    def _forward(self, x: Tensor):
        B, T, D = x.shape
        H, dk = self.num_heads, self.head_dim
        groups = H // self.num_kv_heads

        q_gate = self.q(x).view(B, T, H, dk * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, dk)
        v = self.v(x).view(B, T, self.num_kv_heads, dk)

        q = F.rms_norm(q, (dk,))
        k = F.rms_norm(k, (dk,))
        q_mem, k_mem, v_mem = q, k, v

        y = _rerope_tiled_sdpa(
            q,
            k,
            v,
            rotary=self.rotary,
            groups=groups,
            scale=self.sdpa_scale,
            window=window,
            log_scale=log_scale,
        )
        y = y.reshape(B, T, H * dk)
        y = y * torch.sigmoid(gate.reshape(B, T, -1))
        out = self.proj(y)

        if self.mem is not None:
            out = out + self.mem(
                x,
                q_mem.transpose(1, 2),
                k_mem.repeat_interleave(groups, dim=2).transpose(1, 2),
                v_mem.repeat_interleave(groups, dim=2).transpose(1, 2),
            )
        return out, torch.tensor(0.0, device=x.device)

    return _forward


def install_rotary_mode(model: torch.nn.Module, mode: str, rerope_window: int = TRAIN_LEN // 2) -> None:
    dev = next(model.parameters()).device
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and getattr(attn, "rotary", None) is not None:
            head_dim = attn.head_dim
            if mode in ("rerope", "rerope_log"):
                attn.rotary = ExtendedRotary(head_dim, mode="vanilla").to(dev)
                attn.forward = types.MethodType(
                    _make_rerope_attn_forward(window=rerope_window, log_scale=(mode == "rerope_log")),
                    attn,
                )
            else:
                attn.rotary = ExtendedRotary(head_dim, mode=mode).to(dev)


@torch.no_grad()
def verify_rotary_equivalence(rope_snapshot: str, device: str) -> None:
    """Verify bit-for-bit identity between unmodified train.model.Rotary and ExtendedRotary(vanilla)."""
    scorer = DirectScorer(rope_snapshot, device=device, max_length=256, batch_size=1)
    dev = scorer.device
    x_ids = torch.randint(0, 50256, (1, 128), device=dev, dtype=torch.int32)
    h_orig = scorer._forward_hidden(x_ids).float().clone()

    # Check angular_freq buffer match directly
    for block in scorer.model.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and getattr(attn, "rotary", None) is not None:
            orig_af = attn.rotary.angular_freq.detach().cpu()
            ext_af = ExtendedRotary(attn.head_dim, mode="vanilla").angular_freq.detach().cpu()
            max_af_diff = float((orig_af - ext_af).abs().max().item())
            if max_af_diff != 0.0:
                raise AssertionError(f"angular_freq mismatch: max_af_diff={max_af_diff}")

    install_rotary_mode(scorer.model, mode="vanilla")
    h_van = scorer._forward_hidden(x_ids).float().clone()
    max_h_diff = float((h_orig - h_van).abs().max().item())
    if max_h_diff != 0.0:
        raise AssertionError(f"Vanilla ExtendedRotary hidden mismatch: max_h_diff={max_h_diff}")

    install_rotary_mode(scorer.model, mode="rerope", rerope_window=2048)
    h_rerope_w2048 = scorer._forward_hidden(x_ids).float().clone()
    max_rerope_diff = float((h_orig - h_rerope_w2048).abs().max().item())
    if max_rerope_diff > 5e-2:
        raise AssertionError(f"ReRoPE (T <= window) equivalence check failed: diff={max_rerope_diff}")

    scorer.close()
    print(
        f"[RoPE Baselines] Equivalence verified: vanilla max_diff={max_h_diff:.1e}, "
        f"rerope(T<=w) max_diff={max_rerope_diff:.2e}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vanilla", "pi", "dynamic_ntk", "yarn", "rerope", "rerope_log"],
        choices=["vanilla", "pi", "dynamic_ntk", "yarn", "rerope", "rerope_log"],
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[2048, 8192, 16384, 32768, 65536],
    )
    parser.add_argument("--rerope-window", type=int, default=TRAIN_LEN // 2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "iclr27_rebuttal" / "tables" / "rope_extrapolation_baselines.json",
    )
    args = parser.parse_args()

    rope_snapshot = resolve_rope_snapshot()
    verify_rotary_equivalence(rope_snapshot, args.device)

    bpb_tokens, retrieval_tokens, retrieval_manifest = load_fixtures()
    results = {}
    if args.output.is_file():
        try:
            payload = json.loads(args.output.read_text(encoding="utf-8"))
            if payload.get("rotary_spec") == ROTARY_SPEC_VERSION:
                results = payload.get("results", {})
            else:
                print(
                    f"[RoPE Baselines] Discarding stale {args.output} "
                    f"(rotary_spec={payload.get('rotary_spec')!r} != {ROTARY_SPEC_VERSION!r}).",
                    flush=True,
                )
        except Exception:
            results = {}

    def _save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "rotary_spec": ROTARY_SPEC_VERSION,
                    "base_theta": BASE_THETA,
                    "train_len": TRAIN_LEN,
                    "rerope_window": args.rerope_window,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for mode in args.modes:
        if mode in results:
            print(f"[RoPE Baselines] Mode {mode} already evaluated under {ROTARY_SPEC_VERSION}, skipping.")
            continue
        print(f"\n[RoPE Baselines] Evaluating 9.816B RoPE with mode={mode}...", flush=True)
        t0 = time.time()
        scorer = DirectScorer(rope_snapshot, device=args.device, max_length=262400, batch_size=1)
        install_rotary_mode(scorer.model, mode=mode, rerope_window=args.rerope_window)

        bpb_res = evaluate_bpb(scorer, bpb_tokens, args.lengths)
        ret_res = evaluate_retrieval(scorer, retrieval_tokens, retrieval_manifest, args.lengths)
        scorer.close()

        primary_lens = [l for l in [8192, 16384, 32768, 65536] if l in args.lengths]
        real_exact_mean = (
            sum(ret_res["real"][str(l)]["exact_accuracy_percent"] for l in primary_lens) / len(primary_lens)
            if primary_lens
            else 0.0
        )
        bpb_2k = bpb_res["cross_dataset_mean_bpb"].get("2048", float("nan"))
        real_2k = ret_res["real"].get("2048", {}).get("exact_accuracy_percent", float("nan"))
        print(
            f"  [mode={mode}] 2K Real Exact={real_2k:.2f}% (2K BPB={bpb_2k:.3f}) | "
            f"8K-64K Natural-Text Exact Mean = {real_exact_mean:.2f}% ({time.time() - t0:.1f}s)",
            flush=True,
        )
        results[mode] = {
            "mode": mode,
            "real_text_exact_mean_8k_to_64k": real_exact_mean,
            "bpb": bpb_res,
            "retrieval": ret_res,
        }
        _save()

    print(f"\nSaved RoPE extrapolation baseline evaluations to {args.output}")


if __name__ == "__main__":
    main()
