#!/usr/bin/env python3
"""Evaluate Training-Free RoPE Extrapolation Baselines on the 9.816B RoPE Checkpoint.

Directly addresses Reviewer ihsT (Weakness 1 & Question 1) and Reviewer HE3D (Question 3):
Evaluates the matched 9.816B-token RoPE + gated-delta memory checkpoint under:
1. `vanilla`: Unmodified RoPE (trained at L_train = 2048).
2. `dynamic_ntk`: Context-dependent NTK-aware base scaling
   `base' = base * ((alpha * L / L_train) - (alpha - 1))^(d / (d - 2))` with `alpha = 1`.
3. `yarn`: NTK-by-parts frequency ramp interpolation (beta_fast=32, beta_slow=1)
   combined with YaRN attention logit temperature scaling `sqrt(1/t) = 0.1 * ln(s) + 1`.

Benchmarks evaluated:
- 30-document independent natural-text & synthetic retrieval panel (2K, 8K, 16K, 32K, 64K).
- 18-document fixed-target 3-corpus BPB panel (2K, 8K, 16K, 32K, 64K).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scoring import DirectScorer
from supplementary.pat_feedback.run_e2_experiment import evaluate_bpb, evaluate_retrieval, load_fixtures

ROPE_SNAPSHOT = (
    "/home/sagemaker-user/.cache/huggingface/hub/"
    "models--anon--atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0/"
    "snapshots/de0a2f1c63b0631a040a2d6adac9e2d43fd5c581"
)
TRAIN_LEN = 2048


class ExtendedRotary(torch.nn.Module):
    """Drop-in replacement for `train.model.Rotary` supporting Dynamic-NTK and YaRN."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        train_len: int = TRAIN_LEN,
        mode: str = "vanilla",
        ntk_alpha: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.train_len = int(train_len)
        self.mode = mode
        self.ntk_alpha = float(ntk_alpha)
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        half = dim // 2
        inv_freq = 1.0 / (self.base ** (torch.linspace(0.0, 1.0, half, dtype=torch.float32)))
        self.register_buffer("base_inv_freq", inv_freq, persistent=False)

    def _yarn_inv_freq_and_mscale(self, seq_len: int, device: torch.device) -> tuple[Tensor, float]:
        if seq_len <= self.train_len:
            return self.base_inv_freq.to(device), 1.0
        scale = float(seq_len) / float(self.train_len)
        half = self.dim // 2
        idx = torch.linspace(0.0, 1.0, half, dtype=torch.float32, device=device)
        freq_extrap = 1.0 / (self.base ** idx)
        freq_interp = freq_extrap / scale

        low = max(
            math.floor(half * math.log(self.train_len / (self.beta_fast * 2.0 * math.pi)) / math.log(self.base)),
            0,
        )
        high = min(
            math.ceil(half * math.log(self.train_len / (self.beta_slow * 2.0 * math.pi)) / math.log(self.base)),
            half - 1,
        )
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(half, dtype=torch.float32, device=device) - low) / (high - low), 0.0, 1.0)
        inv_freq = freq_interp * ramp + freq_extrap * (1.0 - ramp)
        mscale = 0.1 * math.log(scale) + 1.0
        return inv_freq, mscale

    def forward(self, x_BTHD: Tensor) -> Tensor:
        seq_len = x_BTHD.size(1)
        device = x_BTHD.device
        mscale = 1.0

        if self.mode == "vanilla" or seq_len <= self.train_len:
            inv_freq = self.base_inv_freq.to(device)
        elif self.mode == "dynamic_ntk":
            scale = (self.ntk_alpha * seq_len / self.train_len) - (self.ntk_alpha - 1.0)
            half = self.dim // 2
            scaled_base = self.base * (scale ** (self.dim / (self.dim - 2.0)))
            inv_freq = 1.0 / (
                scaled_base ** torch.linspace(0.0, 1.0, half, dtype=torch.float32, device=device)
            )
        elif self.mode == "yarn":
            inv_freq, mscale = self._yarn_inv_freq_and_mscale(seq_len, device)
        else:
            raise ValueError(f"Unknown rotary mode: {self.mode}")

        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(pos, inv_freq)[None, :, None, :]
        cos = theta.cos() * mscale
        sin = theta.sin() * mscale
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


def install_rotary_mode(model: torch.nn.Module, mode: str) -> None:
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and getattr(attn, "rotary", None) is not None:
            head_dim = attn.head_dim
            attn.rotary = ExtendedRotary(head_dim, mode=mode).to(next(model.parameters()).device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vanilla", "dynamic_ntk", "yarn"],
        choices=["vanilla", "dynamic_ntk", "yarn"],
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[2048, 8192, 16384, 32768, 65536],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "iclr27_rebuttal" / "tables" / "rope_extrapolation_baselines.json",
    )
    args = parser.parse_args()

    bpb_tokens, retrieval_tokens, retrieval_manifest = load_fixtures()
    results = {}
    if args.output.is_file():
        try:
            results = json.loads(args.output.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            results = {}

    for mode in args.modes:
        if mode in results:
            print(f"[RoPE Baselines] Mode {mode} already evaluated, skipping.")
            continue
        print(f"\n[RoPE Baselines] Evaluating 9.816B RoPE with mode={mode}...", flush=True)
        t0 = time.time()
        scorer = DirectScorer(ROPE_SNAPSHOT, device=args.device, max_length=262400, batch_size=1)
        install_rotary_mode(scorer.model, mode=mode)

        bpb_res = evaluate_bpb(scorer, bpb_tokens, args.lengths)
        ret_res = evaluate_retrieval(scorer, retrieval_tokens, retrieval_manifest, args.lengths)
        scorer.close()

        primary_lens = [l for l in [8192, 16384, 32768, 65536] if l in args.lengths]
        real_exact_mean = (
            sum(ret_res["real"][str(l)]["exact_accuracy_percent"] for l in primary_lens) / len(primary_lens)
            if primary_lens
            else 0.0
        )
        print(
            f"  [mode={mode}] 8K-64K Natural-Text Exact Mean = {real_exact_mean:.2f}% "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        results[mode] = {
            "mode": mode,
            "real_text_exact_mean_8k_to_64k": real_exact_mean,
            "bpb": bpb_res,
            "retrieval": ret_res,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved RoPE extrapolation baseline evaluations to {args.output}")


if __name__ == "__main__":
    main()
