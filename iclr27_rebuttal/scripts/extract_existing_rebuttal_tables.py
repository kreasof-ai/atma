#!/usr/bin/env python3
"""Immediately extract rebuttal tables from existing evaluation JSON records.

Generates in `iclr27_rebuttal/tables/`:
1. `e1_lengthwise_breakdown.md`: Complete 2K-64K natural-text and synthetic exact/token
   retrieval across all 9 E1 runs (3 seeds x 3 arms), plus the exact prompt counts
   explaining why Triplet 1 NoPE and Temp-Softmax both scored 5.69% (41/720 prompts).
2. `e4_lengthwise_breakdown.md`: Complete 2K-256K natural-text and synthetic exact/token
   retrieval across all 6 checkpoints (Primary, Seed 1, Seed 2 x NoPE, Polar).
3. `p90_vs_hl256_comparison.md`: Side-by-side comparison of `p90`, `p99`, `hl-256`,
   and `hl-512` from `gamma_diagnostics/results/all_clamp_sweep_262k.json` answering
   Reviewer HE3D Q2 on why `hl:256` was selected over `p90`.
4. `numerical_reconciliation.md`: Complete reconciliation table for the four numerical
   discrepancies flagged by Reviewers HE3D and ihsT.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "iclr27_rebuttal" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_e1_lengthwise_table() -> None:
    e1_path = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation" / "e1_component_attribution.json"
    data = json.loads(e1_path.read_text(encoding="utf-8"))

    lengths = ["2048", "8192", "16384", "32768", "65536"]

    lines = [
        "# E1 Complete Length-Wise Retrieval Breakdown & Triplet 1 Tie Audit",
        "",
        "## 1. Natural-Text (FinePDFs, 30 Independent Documents, 180 Prompts/Length) Exact Retrieval (%)",
        "",
        "| Triplet (Seed) | Model Arm | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | Exact Hits (8K–64K) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    triplets = [
        ("s1", "Triplet 1 (20270912)"),
        ("s2", "Triplet 2 (20270913)"),
        ("s3", "Triplet 3 (20270914)"),
    ]
    arms = [
        ("nope_memory", "NoPE + Memory"),
        ("temperature_softmax_memory", "Temp-Softmax + Memory"),
        ("polar_memory", "Full Polar + Memory"),
    ]

    for t_key, t_label in triplets:
        for a_key, a_label in arms:
            run_key = f"pat_e1_{t_key}_{a_key}"
            rec = data[run_key]
            real = rec["retrieval"]["real"]
            vals = [real[l]["exact_accuracy_percent"] for l in lengths]
            hits_8k_64k = [round(real[l]["exact_accuracy_percent"] * real[l]["prompts"] / 100.0) for l in lengths[1:]]
            total_hits = sum(hits_8k_64k)
            mean_8k_64k = rec["primary_natural_text_exact_mean_8k_to_64k"]
            hits_str = f"{'+'.join(str(h) for h in hits_8k_64k)} = {total_hits}/720"
            lines.append(
                f"| {t_label} | {a_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
                f"{vals[3]:.2f}% | {vals[4]:.2f}% | **{mean_8k_64k:.2f}%** | `{hits_str}` |"
            )

    lines.extend([
        "",
        "## 2. Why Triplet 1 NoPE and Temp-Softmax Both Averaged `5.69%` on 8K–64K Natural Text",
        "",
        "- Each length evaluates **180 paired prompts** (30 documents $\\times$ 2 tasks $\\times$ 3 depths), totaling **720 prompts** across 8K, 16K, 32K, and 64K.",
        "- In **Triplet 1 (Seed 20270912)**:",
        "  - **NoPE + Memory** achieved `37/180` (`20.56%`) at 8K, `4/180` (`2.22%`) at 16K, `0/180` (`0.00%`) at 32K, and `0/180` (`0.00%`) at 64K $\\implies (37 + 4 + 0 + 0) / 720 = 41 / 720 = 5.6944\\%$.",
        "  - **Temp-Softmax + Memory** achieved `35/180` (`19.44%`) at 8K, `6/180` (`3.33%`) at 16K, `0/180` (`0.00%`) at 32K, and `0/180` (`0.00%`) at 64K $\\implies (35 + 6 + 0 + 0) / 720 = 41 / 720 = 5.6944\\%$.",
        "- The two checkpoints are **not** identical: at 2K natural text, Temp-Softmax scores **72.22%** vs NoPE's **57.22%**; on 64K synthetic haystacks, Temp-Softmax scores **47.22%** vs NoPE's **10.56%**; and on 64K BPB, Temp-Softmax reaches **1.620** vs NoPE's **1.618**. The `5.69%` match on 8K–64K natural text is an integer-sum coincidence (`37+4 = 35+6 = 41` out of 720 prompts) caused by both models collapsing to `0/180` at 32K and 64K.",
        "- **Why the per-seed contrast has a large spread (`+30.56%`, `+31.53%`, `+4.44%`):** As shown in Section 1 above, **Polar is highly stable across all 3 seeds** (`36.25%`, `44.86%`, `48.61%` on 8K–64K; and `32.22%`, `38.33%`, `40.00%` at 64K). The spread in the difference comes from **Temp-Softmax being seed-unstable on natural text**: Temp-Softmax collapses to `0.00%` at 32K–64K in Seeds 1 and 2 (`5.69%` and `13.33%` 8K–64K mean) and only retains 32K–64K recall in Seed 3 (`30.00%` at 64K, `44.17%` mean), where Polar still leads at 32K (`46.67%` vs `36.11%`) and 64K (`40.00%` vs `30.00%`).",
        "",
        "## 3. Synthetic-Haystack Exact Retrieval (%) Across E1 Runs",
        "",
        "| Triplet (Seed) | Model Arm | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | 2K BPB | 64K BPB |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for t_key, t_label in triplets:
        for a_key, a_label in arms:
            run_key = f"pat_e1_{t_key}_{a_key}"
            rec = data[run_key]
            synth = rec["retrieval"]["synthetic"]
            vals = [synth[l]["exact_accuracy_percent"] for l in lengths]
            mean_synth = sum(vals[1:]) / 4.0
            bpb_2k = rec["bpb"]["2048"]["mean_bpb"]
            bpb_64k = rec["bpb"]["65536"]["mean_bpb"]
            lines.append(
                f"| {t_label} | {a_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
                f"{vals[3]:.2f}% | {vals[4]:.2f}% | **{mean_synth:.2f}%** | {bpb_2k:.3f} | {bpb_64k:.3f} |"
            )

    (OUT_DIR / "e1_lengthwise_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_e4_lengthwise_table() -> None:
    e4_path = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation" / "e4_replication_retrieval.json"
    data = json.loads(e4_path.read_text(encoding="utf-8"))["results"]

    lengths = ["2048", "8192", "16384", "32768", "65536", "262144"]
    models = [
        ("primary_nope", "Primary", "NoPE"),
        ("primary_polar", "Primary", "Polar"),
        ("repl_seed1_nope", "Seed 1 (202701)", "NoPE"),
        ("repl_seed1_polar", "Seed 1 (202701)", "Polar"),
        ("repl_seed2_nope", "Seed 2 (202702)", "NoPE"),
        ("repl_seed2_polar", "Seed 2 (202702)", "Polar"),
    ]

    lines = [
        "# E4 Complete Length-Wise Retrieval Breakdown (30 Independent Documents)",
        "",
        "## 1. Natural-Text (FinePDFs, 30 Documents) Exact 5-Token Retrieval (%)",
        "",
        "| Pair | Model | 2K | 8K | 16K | 32K | 64K | 256K | 8K–64K Mean | Paired $\\Delta$ (8K–64K) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for idx, (m_key, pair_label, arch_label) in enumerate(models):
        real = data[m_key]["real"]
        vals = [real[l]["exact_accuracy_percent"] for l in lengths]
        mean_8k_64k = sum(vals[1:5]) / 4.0
        delta_str = "—"
        if arch_label == "Polar":
            nope_key = models[idx - 1][0]
            nope_vals = [data[nope_key]["real"][l]["exact_accuracy_percent"] for l in lengths]
            nope_mean = sum(nope_vals[1:5]) / 4.0
            delta_str = f"**+{mean_8k_64k - nope_mean:.2f}%**"
        lines.append(
            f"| {pair_label} | {arch_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | {vals[5]:.2f}% | **{mean_8k_64k:.2f}%** | {delta_str} |"
        )

    lines.extend([
        "",
        "## 2. Natural-Text (FinePDFs, 30 Documents) Teacher-Forced Token Accuracy (%)",
        "",
        "| Pair | Model | 2K | 8K | 16K | 32K | 64K | 256K | 8K–64K Mean | Paired $\\Delta$ (8K–64K) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for idx, (m_key, pair_label, arch_label) in enumerate(models):
        real = data[m_key]["real"]
        vals = [real[l]["token_accuracy_percent"] for l in lengths]
        mean_8k_64k = sum(vals[1:5]) / 4.0
        delta_str = "—"
        if arch_label == "Polar":
            nope_key = models[idx - 1][0]
            nope_vals = [data[nope_key]["real"][l]["token_accuracy_percent"] for l in lengths]
            nope_mean = sum(nope_vals[1:5]) / 4.0
            delta_str = f"**+{mean_8k_64k - nope_mean:.2f}%**"
        lines.append(
            f"| {pair_label} | {arch_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | {vals[5]:.2f}% | **{mean_8k_64k:.2f}%** | {delta_str} |"
        )

    lines.extend([
        "",
        "## 3. Synthetic-Haystack Exact 5-Token Retrieval (%)",
        "",
        "| Pair | Model | 2K | 8K | 16K | 32K | 64K | 256K | 8K–64K Mean |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for m_key, pair_label, arch_label in models:
        synth = data[m_key]["synthetic"]
        vals = [synth[l]["exact_accuracy_percent"] for l in lengths]
        mean_8k_64k = sum(vals[1:5]) / 4.0
        lines.append(
            f"| {pair_label} | {arch_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | {vals[5]:.2f}% | **{mean_8k_64k:.2f}%** |"
        )

    (OUT_DIR / "e4_lengthwise_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_p90_vs_hl256_table() -> None:
    sweep_path = ROOT / "gamma_diagnostics" / "results" / "all_clamp_sweep_262k.json"
    data = json.loads(sweep_path.read_text(encoding="utf-8"))["checkpoints"]

    ckpts = [
        ("atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (Primary L40S)"),
        ("atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0", "Polar (Primary L40S)"),
        ("atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0", "RoPE (Primary L40S)"),
    ]
    conds = ["baseline", "p99", "p90", "hl-512", "hl-256"]

    lines = [
        "# Why `hl:256` Was Selected Over `p90` (Reviewer HE3D Question 2)",
        "",
        "## 1. Paired Multi-Clamp Sweep Results (`all_clamp_sweep_262k.json`)",
        "",
        "| Checkpoint | Condition | Applied Max Logit | Implied $H_{1/2}$ (tok) | 2K Clean NLL | 256K Clean NLL | 2K Needle CE | 256K Needle CE | 256K Needle Acc (%) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for ck_suffix, ck_label in ckpts:
        matched_key = next(k for k in data if k.endswith(ck_suffix))
        ck_data = data[matched_key]["conditions"]
        for c in conds:
            if c not in ck_data:
                continue
            entry = ck_data[c]
            targets = entry.get("resolved_targets", [])
            if targets:
                max_logit = targets[0]["max_final_logit"]
                hl = targets[0]["max_half_life_tokens"]
                logit_str = f"{max_logit:.4f}"
                hl_str = f"{hl:,.1f}"
            else:
                logit_str = "uncapped"
                hl_str = "uncapped"

            clean = entry["metrics"]["clean"]
            needle = entry["metrics"]["needle"]["by_distance"]
            c_2k = clean["2048"]["loss_nats"]
            c_256k = clean["262144"]["loss_nats"]
            n_2k = needle["2048"]["ce_nats"]
            n_256k = needle["262144"]["ce_nats"]
            acc_256k = needle["262144"]["accuracy_pct"]
            lines.append(
                f"| {ck_label} | `{c}` | {logit_str} | {hl_str} | {c_2k:.3f} | **{c_256k:.3f}** | {n_2k:.3f} | **{n_256k:.3f}** | {acc_256k:.1f}% |"
            )

    lines.extend([
        "",
        "## 2. Three Principled Reasons for Selecting `hl:256` Over `p90`",
        "",
        "1. **Model-Agnostic Physical Timescale vs. Checkpoint-Dependent Quantile:**",
        "   - `p90` clips the outlier head to the 90th percentile of that specific checkpoint's 32 zero-input retention logits ($b_\\gamma + b_{\\text{init}}$). Consequently, `p90` produces a *different* half-life ceiling in every checkpoint (`70.6` tokens in Primary NoPE, `139.2` tokens in Primary Polar, `44.5` tokens in Primary RoPE).",
        "   - `hl:256` ($\\gamma_{\\max} = 2^{-1/256} = 0.997296$, logit $5.9103$) defines a fixed, architecture-wide physical timescale equal to $1/8$ of the $T_{\\text{train}} = 2048$ pretraining window ($2\\times$ the 128-token gated-delta chunk size), independent of checkpoint weight distributions.",
        "2. **`p90` Over-Clips Short-Range Retention (Quadrupling 2K Needle Loss on NoPE), Whereas `hl:256` Preserves Short Context and Beats `p90` at 256K Needle CE:**",
        "   - Because `p90` forces NoPE's cap down to `70.6` tokens (and RoPE's down to `44.5` tokens), it degrades 2K short-context needle cross-entropy on NoPE to **`0.428` nats** ($4.15\\times$ baseline `0.103`) and 2K clean NLL to `2.359` (+0.038 nats drift). In contrast, `hl:256` keeps NoPE 2K needle cross-entropy nearly twice as low (**`0.245` nats**) and 2K clean NLL drift smaller (`2.343`, +0.022 nats), while still achieving **better 256K needle cross-entropy** on both NoPE (`6.051` vs `6.060` nats) and Polar (`1.707` vs `1.713` nats).",
        "3. **Robustness When Multiple Heads Saturate:**",
        "   - Just as `p99` failed completely on NoPE (`H_{1/2} = 484,606` tokens, 256K NLL `12.094`) because the upper quantile remained inside the saturation tail, any quantile-based rule (`p90`) can shift into the unstable regime if more than 3 of the 32 heads drift toward $\\gamma \\to 1$. An absolute half-life ceiling (`hl:256`) guarantees a uniform contraction margin $\\gamma_t \\le 2^{-1/256} < 1$ regardless of how many heads saturate.",
    ])

    (OUT_DIR / "p90_vs_hl256_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_numerical_reconciliation() -> None:
    lines = [
        "# Complete Reconciliation of All Numerical Discrepancies Flagged by Reviewers HE3D & ihsT",
        "",
        "| Discrepancy Flagged | Value A (Location) | Value B (Location) | Exact Root Cause in Codebase | Resolution in Revised Manuscript |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| **1. Polar 128K Prefill / TTFT Latency** | **`1.619 s`** (Table 4, Table 20 `table:serving_detailed_matrix`) | **`0.065 s`** at 128K, **`0.029 s`** at 32K (Table 33 `table:serving_repeatability_e6`) | In `run_e6_serving.py`, Polar ran on the production `LLM` backend (which has automatic 256-token block prefix caching), whereas NoPE/RoPE ran on `BaselineLLM` (no prefix caching). Because `run_e6_serving.py` ran 1 warmup + 10 timed repetitions on the **same engine instance and identical prompt**, repetitions 1–10 hit the prefix cache for the first $131,072 - 256 = 130,816$ tokens and only prefilled the uncached 256-token tail block ($0.06497\\text{ s} \\times 3940.39\\text{ tok/s} = 256.0\\text{ tokens}$). Table 20 (`benchmarks/serving.py`) instantiated a fresh engine with `warmup_samples=0`, measuring true cold-cache 128K prefill (`1.619 s`). | Fixed `run_e6_serving.py` (`fix_e6_serving_benchmark.py`) to clear/reset the prefix cache between timed repetitions; updated Table 33 so all models report cold-cache full-prefix TTFT (`~1.62 s` for Polar at 128K). |",
        "| **2. Primary Polar 256K BPB (Uncapped $\\rightarrow$ Capped)** | **`1.825 -> 1.501`** (Abstract, Table 2, Table 5, Table 30) | **`1.855 -> 1.520`** (Appendix J.4 / E2 text) | Both used the exact same 18 documents and byte counts (`3,636` FinePDFs, `5,039` PG-19, `3,238` Proof-Pile bytes). During the E0 numerical audit (`kernel/polar_triton.py` line 171), the `Q2` participation-ratio denominator floor was tightened from `eps` (`1e-6`) to `1.0e-30` so small running-max scales do not clip `Q2`. E2 (`run_e2_experiment.py`) was run *after* this kernel update (yielding `1.855 -> 1.520`), whereas Tables 2/5/30 were generated from `benchmark_matrix.json` and `run-summary.json` recorded *before* E0 (`1.825 -> 1.501`). (For NoPE, which does not use `polar_triton.py`, both runs produced `8.09775 -> 1.59520`, differing only in 3-decimal rounding `8.097` vs `8.098`.) | Unified all Polar BPB tables to the single post-E0 audited kernel execution path and documented the `+0.030` BPB shift from the `1e-30` `Q2` floor fix. |",
        "| **3. Primary NoPE 256K Likelihood Metrics** | **`10.77` NLL** (Figure 4, Section 6 intro, Table 22) | **`12.003` NLL** (Section 6.2, Table 23) and **`8.097` BPB** (Table 2) | Evaluated on three distinct datasets without explicitly naming the corpus in the prose:<br>• `10.77` nats/token = **FineWeb-Edu validation stream** (`fig_nope_pretrain_loss.pdf`).<br>• `12.003` nats/token = **FinePDFs clean-loss diagnostic stream** (`all_clamp_sweep_262k.json`, Table 23).<br>• `8.097` bits/byte = **18-document 3-corpus fixed-target BPB panel** (8 FinePDFs, 5 PG-19, 5 Proof-Pile). | Explicitly labeled the evaluation corpus (`FineWeb-Edu val NLL`, `FinePDFs clean NLL`, `3-corpus fixed-target BPB`) at every mention in Section 6 and table captions. |",
        "| **4. Primary NoPE 256K BPB in E5 vs E2/Table 2** | **`7.533 -> 1.782` BPB** (Appendix J.4 E5 text) | **`8.098 -> 1.595` BPB** (Table 2, E2) | `run_e5_experiment.py` evaluated a 6-document diagnostic subset (2 docs/corpus, pooled byte denominator) instead of the full 18-document equal-weighted panel. | Re-ran E5 on the full 18-document equal-weighted panel (`fix_e5_full_18doc_panel.py`) for both primary NoPE and primary Polar so E5 matches Table 2 and E2 (`8.098 -> 1.595` and `1.855 -> 1.520`). |",
    ]
    (OUT_DIR / "numerical_reconciliation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_e1_lengthwise_table()
    build_e4_lengthwise_table()
    build_p90_vs_hl256_table()
    build_numerical_reconciliation()
    print(f"Successfully generated all 4 markdown tables in {OUT_DIR}")
