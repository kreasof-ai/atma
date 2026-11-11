#!/usr/bin/env python3
"""Summarize and verify all raw rebuttal artifacts from commit 57435403."""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REB_TABLES = ROOT / "iclr27_rebuttal" / "tables"
PAT_EVAL = ROOT / "supplementary" / "pat_feedback" / "work" / "evaluation"
PAT_RUNS = ROOT / "supplementary" / "pat_feedback" / "work" / "runs"


def inspect_e6():
    print("=" * 90)
    print("1. E6 COLD-CACHE SERVING BENCHMARK (iclr27_rebuttal/tables/e6_fixed_serving.json)")
    print("=" * 90)
    e6 = json.loads((REB_TABLES / "e6_fixed_serving.json").read_text(encoding="utf-8"))["results"]
    for model, mdata in e6.items():
        parts = []
        for l in ["2048", "32768", "131072"]:
            if l not in mdata:
                continue
            c = mdata[l]
            ttft = c["prefill_latency_s"]["median"]
            iqr = c["prefill_latency_s"].get("iqr", 0.0)
            ptok = c["prefill_tokens_per_s"]["median"]
            dlat = c["decode_latency_s_per_token"]["median"]
            dtok = c["decode_tokens_per_s"]["median"]
            if dlat is not None:
                dstr = f"{dlat * 1000:.2f}ms ({dtok:.1f}t/s)"
            else:
                dstr = "Skipped"
            parts.append(f"{int(l)//1024}K: TTFT={ttft:.3f}s (IQR {iqr:.3f}s, {ptok:,.0f}t/s), Dec={dstr}")
        print(f"  {model:20s} | " + " | ".join(parts))


def inspect_e5():
    print("\n" + "=" * 90)
    print("2. E5 FULL 18-DOC PANEL (iclr27_rebuttal/tables/e5_full_18doc_results.json)")
    print("=" * 90)
    e5 = json.loads((REB_TABLES / "e5_full_18doc_results.json").read_text(encoding="utf-8"))["results"]
    for model, mdata in e5.items():
        uncapped_256 = mdata["uncapped"]["cross_dataset_mean_bpb"]["262144"]
        print(f"  Model: {model}")
        for cond, cdata in mdata.items():
            b = cdata["cross_dataset_mean_bpb"]
            hl = cdata["half_life_ceiling"]
            hl_str = f"{hl:.1f}" if hl is not None else "inf"
            recov = uncapped_256 - b["262144"]
            print(
                f"    {cond:22s} (B{cdata['layer']}/H{cdata['head']}, H={hl_str:>9s}): "
                f"2K={b['2048']:.3f} | 64K={b['65536']:.3f} | 256K={b['262144']:.3f} (recovery={recov:+.3f})"
            )


def inspect_e4():
    print("\n" + "=" * 90)
    print("3. E4 PER-DOCUMENT CLUSTER BOOTSTRAP CIs & FREE-GENERATION VERIFICATION")
    print("=" * 90)
    e4 = json.loads((REB_TABLES / "e4_bootstrap_and_freegen.json").read_text(encoding="utf-8"))
    boot = e4["paired_bootstrap_summary"]
    all_doc_diffs = []
    for pair_name in ["primary", "seed1", "seed2"]:
        c = boot[pair_name]["8k_to_64k_real_exact_contrast"]
        print(
            f"  Pair {pair_name:8s} (8K-64K Real Exact Polar-NoPE): "
            f"Mean = +{c['mean']:.2f}% [95% CI: +{c['ci95_low']:.2f}%, +{c['ci95_high']:.2f}%], "
            f"SE = {c['std_err']:.2f}%, Wins/Ties/Losses = {c['positive_docs']}/{c['zero_tie_docs']}/{c['negative_docs']} (of 30 docs)"
        )
        for l in ["2048", "8192", "16384", "32768", "65536"]:
            cl = boot[pair_name]["by_length_real_exact_contrast"][l]
            print(
                f"    L={int(l)//1024:2d}K: Mean={cl['mean']:+6.2f}% "
                f"[95% CI: {cl['ci95_low']:+6.2f}%, {cl['ci95_high']:+6.2f}%] "
                f"(W/T/L: {cl['positive_docs']}/{cl['zero_tie_docs']}/{cl['negative_docs']})"
            )

    # Pooled 3-pair document-cluster bootstrap across the 30 shared documents
    models = e4["models"]
    case_ids = sorted(models["primary_nope"]["real"]["8192"]["per_doc_exact_percent"].keys(), key=int)
    pooled_doc_diffs = []
    for cid in case_ids:
        pair_diffs = []
        for n_k, p_k in [("primary_nope", "primary_polar"), ("repl_seed1_nope", "repl_seed1_polar"), ("repl_seed2_nope", "repl_seed2_polar")]:
            p_m = np.mean([models[p_k]["real"][l]["per_doc_exact_percent"][cid] for l in ["8192", "16384", "32768", "65536"]])
            n_m = np.mean([models[n_k]["real"][l]["per_doc_exact_percent"][cid] for l in ["8192", "16384", "32768", "65536"]])
            pair_diffs.append(p_m - n_m)
        pooled_doc_diffs.append(float(np.mean(pair_diffs)))
    rng = np.random.RandomState(20270915)
    arr = np.array(pooled_doc_diffs)
    idx = rng.randint(0, len(arr), size=(10000, len(arr)))
    bmeans = arr[idx].mean(axis=1)
    print(
        f"  POOLED 3-PAIR MEAN (30 doc clusters): +{arr.mean():.2f}% "
        f"[95% CI: +{np.percentile(bmeans, 2.5):.2f}%, +{np.percentile(bmeans, 97.5):.2f}%], "
        f"Wins/Ties/Losses = {(arr>0).sum()}/{(arr==0).sum()}/{(arr<0).sum()}"
    )

    print("\n  --- Free-Generation vs Teacher-Forced Subset Verification (5 docs x 2 tasks x 3 depths = 30 prompts/length) ---")
    for m_key, mdata in models.items():
        fg_parts = []
        for l in ["2048", "8192", "16384", "32768"]:
            v = mdata["real"][l].get("freegen_subset_verification")
            if v:
                fg_parts.append(
                    f"{int(l)//1024}K: Exact TF={v['tf_exact_percent']:.1f}% / FG={v['freegen_exact_percent']:.1f}% "
                    f"(diff={v['exact_match_discrepancy']:.1f}%), FG_Tok={v['freegen_token_accuracy_percent']:.1f}%"
                )
        print(f"  {m_key:18s} | " + " | ".join(fg_parts))


def inspect_rope_baselines():
    print("\n" + "=" * 90)
    print("4. TRAINING-FREE ROPE EXTRAPOLATION BASELINES ON 9.816B ROPE CHECKPOINT")
    print("=" * 90)
    rope = json.loads((REB_TABLES / "rope_extrapolation_baselines.json").read_text(encoding="utf-8"))["results"]
    for mode, rdata in rope.items():
        bpb = rdata["bpb"]["cross_dataset_mean_bpb"]
        real = rdata["retrieval"]["real"]
        synth = rdata["retrieval"]["synthetic"]
        print(f"  Mode: {mode:12s} | 8K-64K Natural-Text Exact Mean = {rdata['real_text_exact_mean_8k_to_64k']:.2f}%")
        for l in sorted(real.keys(), key=int):
            print(
                f"    {int(l)//1024:3d}K: Real Exact={real[l]['exact_accuracy_percent']:6.2f}% "
                f"(Tok={real[l]['token_accuracy_percent']:6.2f}%) | "
                f"Synth Exact={synth[l]['exact_accuracy_percent']:6.2f}% "
                f"(Tok={synth[l]['token_accuracy_percent']:6.2f}%) | BPB={bpb[l]:.3f}"
            )


def inspect_e1_6arm_suite():
    print("\n" + "=" * 90)
    print("5. COMPLETE 6-ARM x 3-SEED E1 COMPONENT ATTRIBUTION SUITE (18 RUNS AT 2K, 1B TOKENS)")
    print("=" * 90)
    arms = [
        ("rope_memory", "RoPE + Mem"),
        ("nope_memory", "NoPE + Mem"),
        ("temperature_softmax_memory", "Temp-Softmax (Scalable) + Mem"),
        ("polar_direction_only_memory", "Polar Direction-Only + Mem"),
        ("polar_fixed_null_slope_memory", "Polar Fixed-Null-Slope + Mem"),
        ("polar_memory", "Full Polar + Mem"),
    ]
    seeds = [("s1", 20270912), ("s2", 20270913), ("s3", 20270914)]

    arm_means = {a[0]: [] for a in arms}
    for s_tag, seed in seeds:
        print(f"\n  --- Seed {seed} ({s_tag}) ---")
        for a_key, a_label in arms:
            eval_p = PAT_EVAL / f"pat_e1_{s_tag}_{a_key}_eval.json"
            log_p = PAT_RUNS / f"pat_e1_{s_tag}_{a_key}.log"
            ev = json.loads(eval_p.read_text(encoding="utf-8"))
            log_txt = log_p.read_text(encoding="utf-8")
            m_val = re.findall(r"step:1900/1900 val_loss:([0-9.]+)", log_txt)
            val_loss = float(m_val[-1]) if m_val else float("nan")
            r = ev["retrieval"]["real"]
            s = ev["retrieval"]["synthetic"]
            mean_real = ev["primary_natural_text_exact_mean_8k_to_64k"]
            mean_synth = sum(s[l]["exact_accuracy_percent"] for l in ["8192", "16384", "32768", "65536"]) / 4.0
            arm_means[a_key].append(mean_real)
            print(
                f"    {a_label:30s} | ValNLL={val_loss:.4f} | "
                f"Real Exact 2K={r['2048']['exact_accuracy_percent']:5.1f}% "
                f"8K={r['8192']['exact_accuracy_percent']:5.1f}% "
                f"16K={r['16384']['exact_accuracy_percent']:5.1f}% "
                f"32K={r['32768']['exact_accuracy_percent']:5.1f}% "
                f"64K={r['65536']['exact_accuracy_percent']:5.1f}% | "
                f"8K-64K Mean={mean_real:5.2f}% (Synth={mean_synth:5.2f}%) | "
                f"BPB 2K={ev['bpb']['2048']['mean_bpb']:.3f}, 64K={ev['bpb']['65536']['mean_bpb']:.3f}"
            )

    print("\n  --- 3-Seed Cross-Triplet Summary (8K-64K Natural-Text Exact Retrieval) ---")
    full_vals = np.array(arm_means["polar_memory"])
    for a_key, a_label in arms:
        vals = np.array(arm_means[a_key])
        diffs = full_vals - vals
        print(
            f"    {a_label:30s}: Seeds=[{vals[0]:5.2f}%, {vals[1]:5.2f}%, {vals[2]:5.2f}%] | "
            f"Mean={vals.mean():5.2f}% (SD={vals.std(ddof=1):5.2f}%) | "
            f"Full Polar - Arm = {diffs.mean():+6.2f}% [{diffs[0]:+5.2f}%, {diffs[1]:+5.2f}%, {diffs[2]:+5.2f}%]"
        )


def inspect_len8k():
    print("\n" + "=" * 90)
    print("6. 8K PRETRAINING LENGTH CONTROLS (seq_len=8192, 1B TOKENS, SEED 20270912)")
    print("=" * 90)
    arms = [
        ("rope_memory", "RoPE + Mem (8K train)"),
        ("nope_memory", "NoPE + Mem (8K train)"),
        ("temperature_softmax_memory", "Temp-Softmax + Mem (8K train)"),
        ("polar_memory", "Full Polar + Mem (8K train)"),
    ]
    for a_key, a_label in arms:
        eval_p = PAT_EVAL / f"pat_len8k_s1_{a_key}_eval.json"
        log_p = PAT_RUNS / f"pat_len8k_s1_{a_key}.log"
        ev = json.loads(eval_p.read_text(encoding="utf-8"))
        log_txt = log_p.read_text(encoding="utf-8")
        m_val = re.findall(r"step:1900/1900 val_loss:([0-9.]+)", log_txt)
        val_loss = float(m_val[-1]) if m_val else float("nan")
        r = ev["retrieval"]["real"]
        s = ev["retrieval"]["synthetic"]
        mean_real = ev["primary_natural_text_exact_mean_8k_to_64k"]
        extrap_16k_64k = sum(r[l]["exact_accuracy_percent"] for l in ["16384", "32768", "65536"]) / 3.0
        print(
            f"  {a_label:30s} | ValNLL(8K)={val_loss:.4f} | "
            f"Real Exact 2K={r['2048']['exact_accuracy_percent']:5.1f}% "
            f"8K(in-dist)={r['8192']['exact_accuracy_percent']:5.1f}% "
            f"16K={r['16384']['exact_accuracy_percent']:5.1f}% "
            f"32K={r['32768']['exact_accuracy_percent']:5.1f}% "
            f"64K={r['65536']['exact_accuracy_percent']:5.1f}% | "
            f"8K-64K={mean_real:5.2f}%, 16K-64K(extrap)={extrap_16k_64k:5.2f}% | "
            f"BPB 2K={ev['bpb']['2048']['mean_bpb']:.3f}, 64K={ev['bpb']['65536']['mean_bpb']:.3f}"
        )


if __name__ == "__main__":
    inspect_e6()
    inspect_e5()
    inspect_e4()
    inspect_rope_baselines()
    inspect_e1_6arm_suite()
    inspect_len8k()
