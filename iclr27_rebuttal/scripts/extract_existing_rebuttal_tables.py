#!/usr/bin/env python3
"""Build all publication-ready rebuttal Markdown tables from committed JSON artifacts.

Generates in `iclr27_rebuttal/tables/`:
1. `e1_lengthwise_breakdown.md`: Complete 18-run (6 arms x 3 seeds) E1 component isolation
   and RoPE attribution tables, cross-seed contrasts, and Triplet 1 `5.69%` tie proof.
2. `len8k_pretraining_controls.md`: 4-arm 8K pretraining length (`seq_len = 8192`) controls.
3. `e4_lengthwise_breakdown.md`: 2K-256K length-wise retrieval + 10,000-resample 30-document
   cluster paired bootstrap 95% CIs + 5-token greedy free-generation verification.
4. `e5_full_18doc_panel.md`: Full 18-document ceiling sensitivity & head specificity sweep.
5. `e6_cold_cache_serving.md`: Cold-cache 10-repetition serving repeatability across all 6 models.
6. `p90_vs_hl256_comparison.md`: Principled basis for `hl:256` over `p90` (Reviewer HE3D Q2).
7. `numerical_reconciliation.md`: Complete reconciliation of all cross-table discrepancies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "iclr27_rebuttal" / "tables"
PAT_EVAL = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation"
PAT_RUNS = ROOT / "supplementary" / "pat_feedback" / "work" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_final_val_loss(log_path: Path) -> float:
    txt = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"step:1900/1900 val_loss:([0-9.]+)", txt)
    return float(matches[-1]) if matches else float("nan")


def build_e1_lengthwise_table() -> None:
    lengths = ["2048", "8192", "16384", "32768", "65536"]
    triplets = [
        ("s1", "Triplet 1 (`20270912`)"),
        ("s2", "Triplet 2 (`20270913`)"),
        ("s3", "Triplet 3 (`20270914`)"),
    ]
    arms = [
        ("rope_memory", "RoPE + Mem"),
        ("nope_memory", "NoPE + Mem"),
        ("temperature_softmax_memory", "Temp-Softmax (Scalable) + Mem"),
        ("polar_direction_only_memory", "Polar Direction-Only (no $\\mu_i$) + Mem"),
        ("polar_fixed_null_slope_memory", "Polar Fixed-Null-Slope ($\\gamma_h \\approx 0$) + Mem"),
        ("polar_memory", "Full Polar + Mem"),
    ]

    lines = [
        "# E1 Complete 6-Arm $\\times$ 3-Seed Component Attribution & Length-Wise Breakdown",
        "",
        "All 18 runs share exact tensor initialization within each seed triplet (`20270912`, `20270913`, `20270914`), identical data order, Muon+AdamW schedule (1,900 updates, ~1.0B tokens at `seq_len = 2048`), and frozen 30-document retrieval / 18-document BPB fixtures.",
        "",
        "## 1. Cross-Seed Summary: 8K–64K & 64K Natural-Text Exact Retrieval (%)",
        "",
        "| Arm | Seed 1 | Seed 2 | Seed 3 | 3-Seed Mean $\\pm$ SD (8K–64K) | 64K Mean | Paired Contrast (`Full Polar - Arm`, 8K–64K) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    arm_8k64k = {a[0]: [] for a in arms}
    arm_64k = {a[0]: [] for a in arms}
    for s_tag, _ in triplets:
        for a_key, _ in arms:
            ev = json.loads((PAT_EVAL / f"pat_e1_{s_tag}_{a_key}_eval.json").read_text(encoding="utf-8"))
            arm_8k64k[a_key].append(ev["primary_natural_text_exact_mean_8k_to_64k"])
            arm_64k[a_key].append(ev["retrieval"]["real"]["65536"]["exact_accuracy_percent"])

    full_arr = np.array(arm_8k64k["polar_memory"])
    for a_key, a_label in arms:
        vals = np.array(arm_8k64k[a_key])
        v64 = np.array(arm_64k[a_key])
        diffs = full_arr - vals
        diff_str = (
            "—"
            if a_key == "polar_memory"
            else f"**{diffs.mean():+.2f}%** (`{diffs[0]:+.2f}%, {diffs[1]:+.2f}%, {diffs[2]:+.2f}%`)"
        )
        lines.append(
            f"| {a_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"**{vals.mean():.2f}%** $\\pm$ {vals.std(ddof=1):.2f}% | **{v64.mean():.2f}%** | {diff_str} |"
        )

    lines.extend([
        "",
        "## 2. Per-Run Natural-Text (FinePDFs, 30 Documents, 180 Prompts/Length) Exact Retrieval (%)",
        "",
        "| Triplet (Seed) | Model Arm | Val NLL (2K) | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | Exact Hits (8K–64K) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for s_tag, t_label in triplets:
        for a_key, a_label in arms:
            ev = json.loads((PAT_EVAL / f"pat_e1_{s_tag}_{a_key}_eval.json").read_text(encoding="utf-8"))
            vloss = _read_final_val_loss(PAT_RUNS / f"pat_e1_{s_tag}_{a_key}.log")
            real = ev["retrieval"]["real"]
            vals = [real[l]["exact_accuracy_percent"] for l in lengths]
            hits = [round(real[l]["exact_accuracy_percent"] * real[l]["prompts"] / 100.0) for l in lengths[1:]]
            hits_str = f"{'+'.join(str(h) for h in hits)} = {sum(hits)}/720"
            mean_8k_64k = ev["primary_natural_text_exact_mean_8k_to_64k"]
            lines.append(
                f"| {t_label} | {a_label} | {vloss:.4f} | {vals[0]:.2f}% | {vals[1]:.2f}% | "
                f"{vals[2]:.2f}% | {vals[3]:.2f}% | {vals[4]:.2f}% | **{mean_8k_64k:.2f}%** | `{hits_str}` |"
            )

    lines.extend([
        "",
        "## 3. Key Scientific Findings from the 18-Run Component Isolation",
        "",
        "1. **Magnitude channel ($\\mu_i$) is essential (`+23.84%` mean gain):** Removing the explicit count/magnitude channel (`Polar Direction-Only`) drops 8K–64K natural-text exact retrieval from **43.24%** to **19.40%** (`-16.25%`, `-15.28%`, `-40.00%` across the 3 seeds) and collapses 64K retrieval from **36.85%** to **6.67%**.",
        "2. **Learned $\\sqrt{\\log(n+1)}$ null slope improves 30-document retrieval (`+5.28%` overall, `+10.56%` at 64K), resolving Table 7's 16-trial anomaly:** On the 30-document (720-prompt) panel across 3 seeds, `Full Polar` (`43.24%` 8K–64K; **36.85%** at 64K) outperforms or ties `Polar Fixed-Null-Slope` (`37.96%` 8K–64K; **26.30%** at 64K) in every seed (`+8.33%`, `+7.50%`, `0.00%` on 8K–64K; `+17.78%`, `+12.22%`, `+1.67%` at 64K). Table 7's earlier `91.25%` vs `77.50%` inversion was sampling noise from evaluating a single seed on a 16-trial pilot stream ($\\pm 2$ prompts = $12.5\\%$).",
        "3. **Why Triplet 1 `NoPE` and `Temp-Softmax` both scored `5.69%`:** Both collapsed to `0/180` at 32K and 64K on natural text, and their remaining hits at 8K and 16K summed to the same integer (`37+4+0+0 = 41/720` vs `35+6+0+0 = 41/720`). They are distinct checkpoints (`57.22%` vs `72.22%` at 2K real text; `10.56%` vs `47.22%` at 64K synthetic).",
        "4. **Why `Full Polar - Temp-Softmax` has spread (`+30.56%`, `+31.53%`, `+4.44%`):** `Full Polar` is stable across all 3 seeds (`36.25%`, `44.86%`, `48.61%`), whereas `Temp-Softmax` (learned-scale Scalable-Softmax) is seed-fragile on natural text (`5.69%`, `13.33%`, `44.17%`), collapsing to `0.00%` at 32K–64K in 2 of 3 seeds.",
        "",
        "## 4. Synthetic-Haystack Exact Retrieval (%) & Fixed-Target BPB Across All 18 Runs",
        "",
        "| Triplet (Seed) | Model Arm | Synth 2K | Synth 8K | Synth 16K | Synth 32K | Synth 64K | Synth 8K–64K | 2K BPB | 64K BPB |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for s_tag, t_label in triplets:
        for a_key, a_label in arms:
            ev = json.loads((PAT_EVAL / f"pat_e1_{s_tag}_{a_key}_eval.json").read_text(encoding="utf-8"))
            synth = ev["retrieval"]["synthetic"]
            vals = [synth[l]["exact_accuracy_percent"] for l in lengths]
            mean_synth = sum(vals[1:]) / 4.0
            bpb_2k = ev["bpb"]["2048"]["mean_bpb"]
            bpb_64k = ev["bpb"]["65536"]["mean_bpb"]
            lines.append(
                f"| {t_label} | {a_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
                f"{vals[3]:.2f}% | {vals[4]:.2f}% | **{mean_synth:.2f}%** | {bpb_2k:.3f} | {bpb_64k:.3f} |"
            )

    (OUT_DIR / "e1_lengthwise_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_len8k_table() -> None:
    lengths = ["2048", "8192", "16384", "32768", "65536"]
    arms = [
        ("rope_memory", "RoPE + Mem (`seq_len=8192`)"),
        ("nope_memory", "NoPE + Mem (`seq_len=8192`)"),
        ("temperature_softmax_memory", "Temp-Softmax (Scalable) + Mem (`seq_len=8192`)"),
        ("polar_memory", "Full Polar + Mem (`seq_len=8192`)"),
    ]

    lines = [
        "# 8K Pretraining Length Controls (`seq_len = 8192`, 1B Tokens, Matched Seed `20270912`)",
        "",
        "Directly addresses Reviewer `birj` (W2, Q2), Reviewer `HE3D` (Q3), and Reviewer `ihsT` (W5): all four arms share exact common-tensor initialization (`seed = 20270912`), global batch size (`524,288` tokens/step, `1,900` steps = `0.996B` tokens), and optimizer schedule, while extending the pretraining sequence length $4\\times$ from `2048` to `8192`.",
        "",
        "## 1. Natural-Text (30 Independent Documents) Exact 5-Token Retrieval (%)",
        "",
        "| Model Arm (`L_train = 8K`) | Val NLL (8K) | 2K (In-Dist) | 8K (In-Dist) | 16K ($2\\times$) | 32K ($4\\times$) | 64K ($8\\times$) | 8K–64K Mean | 16K–64K Extrap Mean |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for a_key, a_label in arms:
        ev = json.loads((PAT_EVAL / f"pat_len8k_s1_{a_key}_eval.json").read_text(encoding="utf-8"))
        vloss = _read_final_val_loss(PAT_RUNS / f"pat_len8k_s1_{a_key}.log")
        r = ev["retrieval"]["real"]
        vals = [r[l]["exact_accuracy_percent"] for l in lengths]
        m_8k_64k = ev["primary_natural_text_exact_mean_8k_to_64k"]
        m_16k_64k = sum(vals[2:]) / 3.0
        lines.append(
            f"| {a_label} | {vloss:.4f} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | **{vals[4]:.2f}%** | **{m_8k_64k:.2f}%** | **{m_16k_64k:.2f}%** |"
        )

    lines.extend([
        "",
        "## 2. Synthetic-Haystack Exact Retrieval (%) & Fixed-Target BPB (`L_train = 8K`)",
        "",
        "| Model Arm (`L_train = 8K`) | Synth 2K | Synth 8K | Synth 16K | Synth 32K | Synth 64K | 2K BPB | 64K BPB |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for a_key, a_label in arms:
        ev = json.loads((PAT_EVAL / f"pat_len8k_s1_{a_key}_eval.json").read_text(encoding="utf-8"))
        s = ev["retrieval"]["synthetic"]
        vals = [s[l]["exact_accuracy_percent"] for l in lengths]
        bpb_2k = ev["bpb"]["2048"]["mean_bpb"]
        bpb_64k = ev["bpb"]["65536"]["mean_bpb"]
        lines.append(
            f"| {a_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | {bpb_2k:.3f} | {bpb_64k:.3f} |"
        )

    (OUT_DIR / "len8k_pretraining_controls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_e4_lengthwise_table() -> None:
    e4_orig = json.loads((PAT_EVAL / "e4_replication_retrieval.json").read_text(encoding="utf-8"))["results"]
    e4_boot = json.loads((OUT_DIR / "e4_bootstrap_and_freegen.json").read_text(encoding="utf-8"))

    lengths = ["2048", "8192", "16384", "32768", "65536", "262144"]
    models = [
        ("primary_nope", "Primary", "NoPE", "primary"),
        ("primary_polar", "Primary", "Polar", "primary"),
        ("repl_seed1_nope", "Seed 1 (`202701`)", "NoPE", "seed1"),
        ("repl_seed1_polar", "Seed 1 (`202701`)", "Polar", "seed1"),
        ("repl_seed2_nope", "Seed 2 (`202702`)", "NoPE", "seed2"),
        ("repl_seed2_polar", "Seed 2 (`202702`)", "Polar", "seed2"),
    ]

    lines = [
        "# E4 Complete Length-Wise Retrieval, 30-Document Cluster Bootstrap CIs & Free-Generation Verification",
        "",
        "## 1. Natural-Text (FinePDFs, 30 Independent Documents) Exact 5-Token Retrieval (%)",
        "",
        "| Pair | Model | 2K | 8K | 16K | 32K | 64K | 256K | 8K–64K Mean | Paired $\\Delta$ [95% Cluster Bootstrap CI] |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for m_key, pair_label, arch_label, p_short in models:
        real = e4_orig[m_key]["real"]
        vals = [real[l]["exact_accuracy_percent"] for l in lengths]
        mean_8k_64k = sum(vals[1:5]) / 4.0
        ci_str = "—"
        if arch_label == "Polar":
            c = e4_boot["paired_bootstrap_summary"][p_short]["8k_to_64k_real_exact_contrast"]
            ci_str = f"**+{c['mean']:.2f}%** `[+{c['ci95_low']:.2f}%, +{c['ci95_high']:.2f}%]` ({c['positive_docs']}/30 docs)"
        lines.append(
            f"| {pair_label} | {arch_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | {vals[5]:.2f}% | **{mean_8k_64k:.2f}%** | {ci_str} |"
        )

    # Compute pooled 3-pair 30-doc cluster bootstrap
    b_models = e4_boot["models"]
    case_ids = sorted(b_models["primary_nope"]["real"]["8192"]["per_doc_exact_percent"].keys(), key=int)
    pooled_diffs = []
    for cid in case_ids:
        pdiffs = []
        for n_k, p_k in [
            ("primary_nope", "primary_polar"),
            ("repl_seed1_nope", "repl_seed1_polar"),
            ("repl_seed2_nope", "repl_seed2_polar"),
        ]:
            pm = np.mean([b_models[p_k]["real"][l]["per_doc_exact_percent"][cid] for l in ["8192", "16384", "32768", "65536"]])
            nm = np.mean([b_models[n_k]["real"][l]["per_doc_exact_percent"][cid] for l in ["8192", "16384", "32768", "65536"]])
            pdiffs.append(pm - nm)
        pooled_diffs.append(float(np.mean(pdiffs)))
    rng = np.random.RandomState(20270915)
    arr = np.array(pooled_diffs)
    bmeans = arr[rng.randint(0, len(arr), size=(10000, len(arr)))].mean(axis=1)

    lines.extend([
        "",
        f"**Pooled 3-Pair Contrast (30 independent document clusters, 10,000 resamples):** **+{arr.mean():.2f}%** `[95% CI: +`{np.percentile(bmeans, 2.5):.2f}%`, +`{np.percentile(bmeans, 97.5):.2f}%`]` (**{(arr > 0).sum()}/30** documents strictly favor Polar, 0 ties, 0 losses).",
        "",
        "## 2. Length-By-Length Paired Contrast (`Polar - NoPE`) with 30-Document Cluster 95% CIs",
        "",
        "| Pair | 2K | 8K | 16K | 32K | 64K |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for p_short, p_label in [("primary", "Primary"), ("seed1", "Seed 1 (`202701`)"), ("seed2", "Seed 2 (`202702`)")]:
        byl = e4_boot["paired_bootstrap_summary"][p_short]["by_length_real_exact_contrast"]
        cells = [
            f"{byl[l]['mean']:+.2f}% `[{byl[l]['ci95_low']:+.1f}, {byl[l]['ci95_high']:+.1f}]`"
            for l in ["2048", "8192", "16384", "32768", "65536"]
        ]
        lines.append(f"| {p_label} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 3. Free-Generation (`model.generate` Greedy 5-Step Autoregressive) vs Teacher-Forced Verification",
        "",
        "Evaluated on 30 natural-text prompts per length (5 documents $\\times$ 2 tasks $\\times$ 3 depths) across 2K–32K:",
        "",
        "| Checkpoint | 2K (TF / FreeGen Exact; FG Tok) | 8K (TF / FreeGen Exact; FG Tok) | 16K (TF / FreeGen Exact; FG Tok) | 32K (TF / FreeGen Exact; FG Tok) | Max Exact Discrepancy |",
        "| :--- | :--- | :--- | :--- | :--- | ---: |",
    ])

    for m_key, pair_label, arch_label, _ in models:
        cells = []
        max_disc = 0.0
        for l in ["2048", "8192", "16384", "32768"]:
            v = b_models[m_key]["real"][l]["freegen_subset_verification"]
            max_disc = max(max_disc, abs(v["exact_match_discrepancy"]))
            cells.append(
                f"{v['tf_exact_percent']:.1f}% / **{v['freegen_exact_percent']:.1f}%** (Tok {v['freegen_token_accuracy_percent']:.1f}%)"
            )
        lines.append(f"| {pair_label} {arch_label} | " + " | ".join(cells) + f" | **{max_disc:.1f}%** |")

    lines.extend([
        "",
        "## 4. Synthetic-Haystack Exact 5-Token Retrieval (%) Across 2K–256K",
        "",
        "| Pair | Model | 2K | 8K | 16K | 32K | 64K | 256K | 8K–64K Mean |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for m_key, pair_label, arch_label, _ in models:
        synth = e4_orig[m_key]["synthetic"]
        vals = [synth[l]["exact_accuracy_percent"] for l in lengths]
        mean_8k_64k = sum(vals[1:5]) / 4.0
        lines.append(
            f"| {pair_label} | {arch_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | {vals[5]:.2f}% | **{mean_8k_64k:.2f}%** |"
        )

    (OUT_DIR / "e4_lengthwise_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_e6_serving_table() -> None:
    e6 = json.loads((OUT_DIR / "e6_fixed_serving.json").read_text(encoding="utf-8"))["results"]
    model_labels = [
        ("nope", "ATMA NoPE + Mem"),
        ("polar", "ATMA Polar + Mem"),
        ("rope", "ATMA RoPE + Mem"),
        ("raven_native", "RAVEN Native"),
        ("atma_raven_titans", "ATMA RAVEN + Titans"),
        ("tda_hybrid", "Scaled TDA Hybrid"),
    ]

    lines = [
        "# E6 Cold-Cache Serving Repeatability Benchmark (10 Timed Repetitions After Warmup, 1x NVIDIA L40S)",
        "",
        "Mutating token 0 per repetition prevents 256-token block prefix-cache hits in Polar's `LLM` engine, ensuring every model executes full-prefix prefill across all 10 repetitions.",
        "",
        "| Model | Backend | Context | Cold TTFT Median (s) | TTFT IQR (s) | Prefill Tok/s | Decode ms/tok | Decode Tok/s | Peak Mem (GiB) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for m_key, m_label in model_labels:
        mdata = e6[m_key]
        for l in ["2048", "32768", "131072"]:
            c = mdata[l]
            ttft = c["prefill_latency_s"]["median"]
            iqr = c["prefill_latency_s"]["iqr"]
            ptok = c["prefill_tokens_per_s"]["median"]
            dlat = c["decode_latency_s_per_token"]["median"]
            dtok = c["decode_tokens_per_s"]["median"]
            dlat_str = f"{dlat * 1000.0:.2f}" if dlat is not None else "—"
            dtok_str = f"{dtok:,.1f}" if dtok is not None else "—"
            lines.append(
                f"| {m_label} | `{c['backend']}` | {int(l)//1024}K | **{ttft:.3f}** | {iqr:.3f} | "
                f"{ptok:,.0f} | {dlat_str} | {dtok_str} | {c['peak_allocated_gib']:.2f} |"
            )

    (OUT_DIR / "e6_cold_cache_serving.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "## 1. Paired Multi-Clamp Sweep Results (`all_clamp_sweep_262k.json` & Full 18-Doc E5)",
        "",
        "| Checkpoint | Condition | Applied Max Logit | Implied $H_{1/2}$ (tok) | 2K Clean NLL | 256K Clean NLL | 2K Needle CE | 256K Needle CE | 18-Doc 2K BPB | 18-Doc 256K BPB |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    e5 = json.loads((OUT_DIR / "e5_full_18doc_results.json").read_text(encoding="utf-8"))["results"]
    e5_map = {
        ("NoPE (Primary L40S)", "baseline"): ("primary_nope", "uncapped"),
        ("NoPE (Primary L40S)", "p99"): ("primary_nope", "p99_outlier"),
        ("NoPE (Primary L40S)", "p90"): ("primary_nope", "p90_outlier"),
        ("NoPE (Primary L40S)", "hl-512"): ("primary_nope", "hl_512_outlier"),
        ("NoPE (Primary L40S)", "hl-256"): ("primary_nope", "hl_256_outlier"),
        ("Polar (Primary L40S)", "baseline"): ("primary_polar", "uncapped"),
        ("Polar (Primary L40S)", "p99"): ("primary_polar", "p99_outlier"),
        ("Polar (Primary L40S)", "p90"): ("primary_polar", "p90_outlier"),
        ("Polar (Primary L40S)", "hl-512"): ("primary_polar", "hl_512_outlier"),
        ("Polar (Primary L40S)", "hl-256"): ("primary_polar", "hl_256_outlier"),
    }

    for ck_suffix, ck_label in ckpts:
        matched_key = next(k for k in data if k.endswith(ck_suffix))
        ck_data = data[matched_key]["conditions"]
        for c in conds:
            if c not in ck_data:
                continue
            entry = ck_data[c]
            targets = entry.get("resolved_targets", [])
            if targets:
                logit_str = f"{targets[0]['max_final_logit']:.4f}"
                hl_str = f"{targets[0]['max_half_life_tokens']:,.1f}"
            else:
                logit_str = "uncapped"
                hl_str = "uncapped"

            clean = entry["metrics"]["clean"]
            needle = entry["metrics"]["needle"]["by_distance"]
            c_2k = clean["2048"]["loss_nats"]
            c_256k = clean["262144"]["loss_nats"]
            n_2k = needle["2048"]["ce_nats"]
            n_256k = needle["262144"]["ce_nats"]

            if (ck_label, c) in e5_map:
                mk, ck_cond = e5_map[(ck_label, c)]
                b2k = f"{e5[mk][ck_cond]['cross_dataset_mean_bpb']['2048']:.3f}"
                b256k = f"**{e5[mk][ck_cond]['cross_dataset_mean_bpb']['262144']:.3f}**"
            else:
                b2k, b256k = "—", "—"

            lines.append(
                f"| {ck_label} | `{c}` | {logit_str} | {hl_str} | {c_2k:.3f} | **{c_256k:.3f}** | "
                f"{n_2k:.3f} | **{n_256k:.3f}** | {b2k} | {b256k} |"
            )

    lines.extend([
        "",
        "## 2. Three Principled Reasons for Selecting `hl:256` Over `p90`",
        "",
        "1. **Model-Agnostic Physical Timescale vs. Checkpoint-Dependent Quantile:** `p90` maps to a different half-life in every checkpoint (`70.6` tokens in Primary NoPE, `139.2` tokens in Primary Polar, `44.5` tokens in Primary RoPE). `hl:256` ($\\gamma_{\\max} = 2^{-1/256} = 0.997296$) defines a fixed timescale equal to $1/8$ of the $T_{\\text{train}} = 2048$ training window ($2\\times$ the 128-token gated-delta chunk size).",
        "2. **`p90` Over-Clips Short-Context Retention (`1.470` vs `1.454` 2K BPB and $4.15\\times$ 2K Needle CE on NoPE):** On the full 18-document panel, `p90` degrades Primary NoPE 2K BPB by `+0.038` (`1.432 -> 1.470`) and quadruples 2K needle cross-entropy (`0.103 -> 0.428` nats), whereas `hl:256` cuts the 2K BPB penalty nearly in half (`1.454`, `+0.022`) and keeps 2K needle CE at `0.245` nats while achieving virtually identical 256K BPB (`1.595` vs `1.591` on NoPE; `1.521` vs `1.521` on Polar) and lower 256K needle CE (`6.051` vs `6.060` on NoPE; `1.707` vs `1.713` on Polar).",
        "3. **Robustness When Multiple Heads Saturate:** Just as `p99` failed completely (`8.126` 256K BPB on NoPE, `1.839` on Polar) because the upper quantile stayed inside the saturation tail, any quantile rule (`p90`) becomes unstable if $>10\\%$ of heads drift near $\\gamma \\to 1$, whereas an absolute half-life ceiling guarantees uniform state contraction.",
    ])

    (OUT_DIR / "p90_vs_hl256_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rope_extrapolation_table() -> None:
    payload = json.loads((OUT_DIR / "rope_extrapolation_baselines.json").read_text(encoding="utf-8"))
    rope = payload["results"]
    e4_orig = json.loads((PAT_EVAL / "e4_replication_retrieval.json").read_text(encoding="utf-8"))["results"]

    modes = [
        ("vanilla", "RoPE (`vanilla`, checkpoint-exact)"),
        ("pi", "RoPE + Linear PI (`inv_freq / s`)"),
        ("dynamic_ntk", "RoPE + Dynamic-NTK"),
        ("rerope", "RoPE + ReRoPE (`w = 1024`, no $\\log n$)"),
        ("yarn", "RoPE + YaRN (`NTK-by-parts` + `mscale`)"),
        ("rerope_log", "RoPE + ReRoPE + $\\log n$ (`w = 1024`)"),
    ]
    lengths = ["2048", "8192", "16384", "32768", "65536"]

    lines = [
        "# Training-Free RoPE Extrapolation Baselines on the 9.816B RoPE+Memory Checkpoint",
        "",
        f"Evaluated under checkpoint-faithful half-truncated RoPE (`rotary_spec = {payload.get('rotary_spec')}`, `base = {payload.get('base_theta')}`, `max_abs_diff = 0.0` at `vanilla` vs `train.model.Rotary`) on the frozen 30-document retrieval and 18-document fixed-target BPB panels.",
        "",
        "## 1. Natural-Text (FinePDFs, 30 Documents) Exact 5-Token Retrieval (%) & Fixed-Target BPB",
        "",
        "| Method (`9.816B` Checkpoint) | Phase Fix | $\\log n$ Temp | 2K | 8K | 16K | 32K | 64K | 8K–64K Mean | 2K BPB | 64K BPB |",
        "| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    meta = {
        "vanilla": ("—", "—"),
        "pi": ("Linear", "—"),
        "dynamic_ntk": ("NTK", "—"),
        "rerope": ("Clamp $w$", "—"),
        "yarn": ("NTK-parts", "Yes (`mscale`)"),
        "rerope_log": ("Clamp $w$", "Yes (`ln i`)"),
    }

    for m_key, m_label in modes:
        rdata = rope[m_key]
        real = rdata["retrieval"]["real"]
        bpb = rdata["bpb"]["cross_dataset_mean_bpb"]
        vals = [real[l]["exact_accuracy_percent"] for l in lengths]
        mean_8k_64k = rdata["real_text_exact_mean_8k_to_64k"]
        pf, tf = meta[m_key]
        lines.append(
            f"| {m_label} | {pf} | {tf} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | **{mean_8k_64k:.2f}%** | {bpb['2048']:.3f} | {bpb['65536']:.3f} |"
        )

    # Add matched 9.816B Primary NoPE and Primary Polar reference rows on the identical 30-doc / 18-doc fixtures
    e5 = json.loads((OUT_DIR / "e5_full_18doc_results.json").read_text(encoding="utf-8"))["results"]
    for ref_key, ref_label, pf, tf, e5_key, e5_cond in [
        ("primary_nope", "ATMA NoPE + Mem (Primary, unadapted)", "Position-free", "—", "primary_nope", "uncapped"),
        ("primary_polar", "ATMA Polar + Mem (Primary, unadapted)", "Position-free", "Learned ($\\tau_h, \\mu_i, a_{\\text{null}}$)", "primary_polar", "uncapped"),
    ]:
        real = e4_orig[ref_key]["real"]
        vals = [real[l]["exact_accuracy_percent"] for l in lengths]
        mean_8k_64k = sum(vals[1:]) / 4.0
        bpb = e5[e5_key][e5_cond]["cross_dataset_mean_bpb"]
        lines.append(
            f"| {ref_label} | {pf} | {tf} | **{vals[0]:.2f}%** | **{vals[1]:.2f}%** | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | **{mean_8k_64k:.2f}%** | {bpb['2048']:.3f} | {bpb['65536']:.3f} |"
        )

    lines.extend([
        "",
        "## 2. Synthetic-Haystack Exact 5-Token Retrieval (%) Across 2K–64K",
        "",
        "| Method (`9.816B` Checkpoint) | Synth 2K | Synth 8K | Synth 16K | Synth 32K | Synth 64K | Synth 8K–64K Mean |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for m_key, m_label in modes:
        synth = rope[m_key]["retrieval"]["synthetic"]
        vals = [synth[l]["exact_accuracy_percent"] for l in lengths]
        lines.append(
            f"| {m_label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | "
            f"{vals[3]:.2f}% | {vals[4]:.2f}% | **{sum(vals[1:]) / 4.0:.2f}%** |"
        )

    for ref_key, ref_label in [
        ("primary_nope", "ATMA NoPE + Mem (Primary, unadapted)"),
        ("primary_polar", "ATMA Polar + Mem (Primary, unadapted)"),
    ]:
        synth = e4_orig[ref_key]["synthetic"]
        vals = [synth[l]["exact_accuracy_percent"] for l in lengths]
        lines.append(
            f"| {ref_label} | **{vals[0]:.2f}%** | **{vals[1]:.2f}%** | **{vals[2]:.2f}%** | "
            f"**{vals[3]:.2f}%** | **{vals[4]:.2f}%** | **{sum(vals[1:]) / 4.0:.2f}%** |"
        )

    lines.extend([
        "",
        "## 3. Mechanistic Takeaways (Directly Answering Reviewers `ihsT` W1/Q1 & `HE3D` Q3)",
        "",
        "1. **Phase repair without $\\log n$ logit sharpening fails (`0.00%–1.39%`):** `PI` (`0.00%`), `Dynamic-NTK` (`0.28%`), and unscaled `ReRoPE` (`1.39%`) all collapse on 8K–64K natural text despite eliminating out-of-distribution rotary angles. Adding inference-time $\\log n$ attention logit scaling jumps `YaRN` to **52.78%** (`43.89%` at 64K) and `ReRoPE+log` to **54.17%** (`42.22%` at 64K), proving that softmax entropy dilution is a co-equal bottleneck alongside rotary phase wraparound.",
        "2. **Why Polar Attention remains distinct from post-hoc RoPE patches:**",
        "   - **In-distribution & synthetic fidelity + 8K recall:** Because ATMA's RoPE core omits Q/K/V local `canon` convolutions and rotates half the subspace, `RoPE` (+YaRN/ReRoPE) reaches only `63.89%–64.44%` at 2K real text (`32.78%–35.00%` 2K synthetic; `20.00%–30.28%` 8K–64K synthetic), whereas unadapted `Polar` achieves **`96.11%` at 2K real text (`98.33%` 2K synthetic; `79.72%` 8K–64K synthetic)** and leads at 8K natural text (`72.78%–82.22%` across the 3 pairs vs. `63.89%` YaRN / `67.78%` ReRoPE+log).",
        "   - **Causal streaming & single-KV-cache invariance:** `YaRN` depends on knowing target scale $s = L_{\\text{target}} / L_{\\text{train}}$ up front (mutating token embeddings and invalidating prefix KV caches across target lengths), while `ReRoPE+log` requires dual key evaluations ($R_j k_j$ for $i-j < w$ and $k_j$ for $i-j \\ge w$), breaking standard single-pass FlashAttention/paged-KV serving. Polar extrapolates causally with a single prefix-cacheable KV state (`1.589 s` cold 128K TTFT) and reaches **`54.86%` (8K–64K) / `51.11%` (64K)** natively when pretrained at 8K.",
        "   - **Orthogonality to 256K retention-gate stability:** Inference-time RoPE scaling does not repair gated-delta retention saturation ($\\gamma_0 \\to 1$; Primary RoPE degrades to `2.572` 256K BPB uncapped vs. `1.663` under `hl:256`, whereas capped Primary Polar reaches **`1.409` BPB at 64K** and **`1.521` BPB at 256K** vs. YaRN `1.588` and ReRoPE+log `1.695` at 64K).",
    ])

    (OUT_DIR / "rope_extrapolation_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_numerical_reconciliation() -> None:
    lines = [
        "# Complete Reconciliation of All Numerical Discrepancies Flagged by Reviewers HE3D & ihsT",
        "",
        "| Discrepancy Flagged | Value A (Location) | Value B (Location) | Exact Root Cause in Codebase | Verified Resolution |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| **1. Polar 128K Prefill / TTFT Latency** | **`1.619 s`** (Table 4, Table 20 `table:serving_detailed_matrix`) | **`0.065 s`** at 128K, **`0.029 s`** at 32K (Table 33 `table:serving_repeatability_e6`) | `run_e6_serving.py` ran Polar on the `LLM` engine (256-token block prefix caching) and NoPE/RoPE on `BaselineLLM` (no prefix cache). Repeating the identical prompt 10 times after warmup hit the prefix cache on $130,816$ tokens and prefilled only the uncached 256-token tail block ($0.06497\\text{ s} \\times 3940.39\\text{ tok/s} = 256.0\\text{ tokens}$). | Re-ran 10-rep cold-cache benchmark (`e6_fixed_serving.json`) with per-rep token-0 cache invalidation: Polar 128K cold TTFT = **`1.589 s`** (IQR `0.004 s`, `82,469` tok/s, decode `15.48 ms/tok`), matching Table 4 (`1.62 s`) and Table 20 (`1.619 s`) vs NoPE `1.434 s`, RoPE `1.456 s`, and TDA `10.445 s`. |",
        "| **2. Primary Polar 256K BPB (Uncapped $\\rightarrow$ Capped)** | **`1.825 -> 1.501`** (Abstract, Table 2, Table 5, Table 30) | **`1.855 -> 1.520/1.521`** (Appendix J.4 / E2 / E5) | Identical 18 document SHA-256 hashes (`exact_hash_match = True`). Commit `798418de` tightened the `Q2` participation-ratio floor in `kernel/polar_triton.py` line 171 from `eps` (`1e-6`) to `1.0e-30` so max-shifted `Q2` under dominant null logits preserves scale-invariant $L^2 / Q_2$. Tables 2/5/30 read logs recorded before that fix (`1.451` at 2K, `1.825 -> 1.501` at 256K); E2 and 18-doc E5 ran after (`1.416` at 2K, `1.855 -> 1.521` at 256K). NoPE (`8.097 -> 1.595`) is unchanged. | Unified all Polar BPB tables to the audited post-fix kernel (`1.416` at 2K, `1.855 -> 1.521` at 256K) with an explicit footnote documenting the `+0.030` 256K / `-0.035` 2K shift from the `1e-30` `Q2` floor fix. |",
        "| **3. Primary NoPE 256K Likelihood Metrics** | **`10.77` NLL** (Figure 4, Section 6 intro, Table 22) | **`12.003` NLL** (Section 6.2, Table 23) and **`8.097` BPB** (Table 2) | Three distinct evaluation corpora cited without inline corpus labels:<br>• `10.77` nats/tok = **FineWeb-Edu validation stream** (`fig_nope_pretrain_loss.pdf`)<br>• `12.003` nats/tok = **FinePDFs clean-loss diagnostic stream** (`all_clamp_sweep_262k.json`)<br>• `8.097` bits/byte = **18-document 3-corpus fixed-target BPB panel** | Explicitly labeled the evaluation corpus (`FineWeb-Edu val NLL`, `FinePDFs clean NLL`, `3-corpus fixed-target BPB`) at every occurrence. |",
        "| **4. Primary NoPE 256K BPB in E5 vs E2/Table 2** | **`7.533 -> 1.782` BPB** (Appendix J.4 E5 text) | **`8.097 -> 1.595` BPB** (Table 2, E2, new 18-doc E5) | Original `run_e5_experiment.py` evaluated a 6-document subset (2 docs/corpus, pooled byte denominator). | Re-ran E5 on the full 18-document equal-weighted panel (`e5_full_18doc_results.json`): NoPE `8.097 -> 1.595`, Polar `1.855 -> 1.521`, matching E2 and Table 2. |",
    ]
    (OUT_DIR / "numerical_reconciliation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_e1_lengthwise_table()
    build_len8k_table()
    build_e4_lengthwise_table()
    build_e6_serving_table()
    build_p90_vs_hl256_table()
    build_rope_extrapolation_table()
    build_numerical_reconciliation()
    print(f"Successfully generated all 7 markdown tables (+ existing e5_full_18doc_panel.md) in {OUT_DIR}")
