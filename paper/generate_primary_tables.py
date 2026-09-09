"""Regenerate primary/replication and untouched retrieval tables from full logs.

Run from any directory with Python; --check verifies the committed tables.
Smoke runs are excluded and complete task/depth coverage is required.
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from tda_data import load_tda

from re_evaluation_data import (
    ALL_MODELS, DATASETS, DEPTHS, LENGTHS, ROOT,
    baseline_babilong, baseline_haystack_retrieval,
    baseline_longdoc, baseline_retrieval, mean_longdoc,
    capped_retrieval, capped_babilong, capped_longdoc,
)

LABELS = {"nope": "NoPE", "polar": "Polar", "rope": "RoPE",
          "raven_native": "Raven Native", "atma_raven_titans": "Atma-Raven-Titans", "tda_hybrid": "TDA hybrid"}
ORDER = ("nope", "polar", "rope", "atma_raven_titans", "raven_native")


def table(columns, header, rows, caption, label):
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header,
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}",
    ])


def primary_endpoints():
    token = baseline_retrieval(models=ALL_MODELS)
    exact = baseline_retrieval("exact_match", models=ALL_MODELS)
    bpb = mean_longdoc(baseline_longdoc(models=ALL_MODELS))
    babi = baseline_babilong(models=ALL_MODELS)
    capped_token = capped_retrieval()
    capped_exact = capped_retrieval("exact_match")
    capped_bpb = mean_longdoc(capped_longdoc())
    capped_babi = capped_babilong()
    rows = []
    for model in ORDER:
        if model == "atma_raven_titans":
            rows.append(r"\midrule")
        rows.append(
            f"{LABELS[model]} & {token[model]['256k']:.1f} & {exact[model]['256k']:.1f} & "
            f"{babi[model]['256k']:.0f} & {bpb[model]['2k']:.3f} & {bpb[model]['256k']:.3f}" + r" \\")
        if model in ("nope", "polar"):
            rows.append(f"{LABELS[model]} + cap & {capped_token[model]['256k']:.1f} & "
                        f"{capped_exact[model]['256k']:.1f} & {capped_babi[model]['256k']:.0f} & "
                        f"{capped_bpb[model]['2k']:.3f} & {capped_bpb[model]['256k']:.3f}" + r" \\")
    tda = load_tda()
    rows += [f"TDA hybrid & {tda['retrieval']['token_accuracy']['overall']['256k']:.1f} & "
             f"{tda['retrieval']['exact_match']['overall']['256k']:.1f} & {tda['babi']['256k']:.0f} & "
             f"{tda['bpb']['mean']['2k']:.3f} & {tda['bpb']['mean']['256k']:.3f}" + r" \\"]
    return table("lrrrrr",
        r"\textbf{Model} & \multicolumn{2}{c}{\textbf{Retrieval 256K (\%)}} & \textbf{BABI 256K} & \multicolumn{2}{c}{\textbf{BPB}} \\"
        "\n" + r" & Token & Exact & (\%) & 2K & 256K \\", rows,
        r"Primary endpoints with paired inference-only 256-token half-life caps for NoPE and Polar. Unmarked rows are untouched; caps do not denote retraining. Raven and TDA hybrid use separate AdamW recipes. Retrieval averages tasks, suites, and depths; exact requires all five target tokens to be correct. BABILong is macro exact match after adaptation. BPB averages three fixed-target datasets (lower is better).",
        "tab:endpoints")


def replication_endpoints():
    path = ROOT / "supplementary/robustness/work/evaluation/replication/run-summary.json"
    jobs = json.loads(path.read_text(encoding="utf-8"))["results"]
    indexed = {(j["family"], j["model"], j["suite"]): j["result"]
               for j in jobs if j["condition"] == "baseline" and j["status"] == "complete"}
    token = baseline_retrieval()
    exact = baseline_retrieval("exact_match")
    bpb = mean_longdoc(baseline_longdoc())
    rows = []
    for model in ("nope", "polar"):
        if rows:
            rows.append(r"\midrule")
        rows.append(f"{LABELS[model]} & Primary (mbs16) & {token[model]['256k']:.2f} & "
                    f"{exact[model]['256k']:.2f} & {bpb[model]['256k']:.3f}" + r" \\")
        for seed_index, seed in ((1, 202701), (2, 202702)):
            name = f"repl_seed{seed_index}_{model}"
            scores = {}
            for metric in ("results", "exact_results"):
                scores[metric] = mean(
                    indexed[("retrieval", name, suite)][metric][task]["256k"][depth]
                    for suite in ("synthetic", "real")
                    for task in ("niah", "passkey") for depth in DEPTHS)
            loss = mean(indexed[("longdoc", name, "fixed-target")]["results"][d]
                        ["lengths"]["256k"]["bits_per_byte"] for d in DATASETS)
            rows.append(f"{LABELS[model]} & {seed} (mbs4) & {scores['results']:.2f} & "
                        f"{scores['exact_results']:.2f} & {loss:.3f}" + r" \\")
    return table("llrrr",
        r"\textbf{Model} & \textbf{Training run} & \multicolumn{2}{c}{\textbf{Retrieval 256K (\%)}} & \textbf{BPB 256K} \\"
        "\n" + r" & & Token & Exact & \\", rows,
        r"Untouched primary and fresh paired runs at 256K. All use 9.816B tokens and length 2K on L40S. Each fresh seed pairs NoPE and Polar; the primary checkpoints have no recorded common initialization seed and use a larger microbatch. These rows describe training-run variability, not an isolated seed effect. Exact successes at 256K are synthetic.",
        "tab:replication_main")


def retrieval_breakdown(metric, label):
    values = baseline_haystack_retrieval(metric)
    values["tda_hybrid"] = load_tda()["retrieval"][metric]
    rows = []
    for model in ("polar", "nope", "rope", "atma_raven_titans", "raven_native", "tda_hybrid"):
        for suite in ("synthetic", "real"):
            name = LABELS[model] if suite == "synthetic" else ""
            haystack = "Synthetic" if suite == "synthetic" else "FinePDFs"
            rows.append(f"{name} & {haystack} & " + " & ".join(
                f"{values[model][suite][length]:.1f}" for length in LENGTHS) + r" \\")
    metric_text = "target-token" if metric == "token_accuracy" else "exact five-token"
    result = table("llrrrrrrrr",
        r"\textbf{Model} & \textbf{Haystack} & " + " & ".join(
            rf"\textbf{{{length.upper()}}}" for length in LENGTHS) + r" \\", rows,
        rf"Untouched teacher-forced {metric_text} retrieval accuracy (\%). Each entry averages two tasks and three depths, with 50 trials per cell. Only full evaluations enter these means; smoke runs are excluded.", label)
    # Full eight-length matrices belong in the appendix; retain their established fit.
    return result.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{%" + "\n" + r"\begin{tabular}", 1).replace(
        r"\end{tabular}", r"\end{tabular}}", 1)


def replace_table(source, label, replacement):
    pattern = r"\\begin\{table\}(?:\[[^\]]*\])?(?:(?!\\begin\{table\}).)*?\\label\{" + re.escape(label) + r"\}.*?\\end\{table\}"
    matches = list(re.finditer(pattern, source, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"expected one table {label}, found {len(matches)}")
    match = matches[0]
    return source[:match.start()] + replacement + source[match.end():]


def reference_tables():
    """Full likelihood and adapted reasoning curves, including paired caps."""
    d = load_tda()
    bpb = baseline_longdoc(models=ALL_MODELS)
    babi = baseline_babilong(models=ALL_MODELS)
    bpb["tda_hybrid"] = d["bpb"]
    babi["tda_hybrid"] = d["babi"]
    cbpb, cbabi = capped_longdoc(), capped_babilong()
    order = []
    for model in ORDER + ("tda_hybrid",):
        order.append((LABELS[model], bpb[model], babi[model]))
        if model in ("nope", "polar"):
            order.append((LABELS[model] + " + cap", cbpb[model], cbabi[model]))
    def matrix(rows, caption, label):
        t = table("llrrrrrrrr", r"\textbf{Model} & \textbf{Dataset} & " + " & ".join(l.upper() for l in LENGTHS) + r" \\", rows, caption, label)
        return t.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{%" + "\n" + r"\begin{tabular}", 1).replace(r"\end{tabular}", r"\end{tabular}}", 1)
    rows = []
    for name, bp, _ in order:
        for i, (key, title) in enumerate((("finepdfs", "FinePDFs"), ("pg19", "PG-19"), ("proof_pile", "Proof-Pile"))):
            rows.append((name if i == 0 else "") + " & " + title + " & " + " & ".join(f"{bp[key][l]:.3f}" for l in LENGTHS) + r" \\")
    loss = matrix(rows, r"Fixed-target BPB on the same 8 FinePDFs, 5 PG-19, and 5 Proof-Pile documents. Unmarked rows are untouched. Adjacent NoPE/Polar rows apply the inference-only 256-token half-life cap. Raven and TDA use separate AdamW recipes; lower is better.", "tab:all_bpb")
    rows = [name + " & " + " & ".join(f"{ba[l]:.0f}" for l in LENGTHS) + r" \\" for name, _, ba in order]
    reasoning = table("lrrrrrrrr", r"\textbf{Model} & " + " & ".join(l.upper() for l in LENGTHS) + r" \\", rows, r"Adapted BABILong macro exact match (\%) under the common short-context adaptation protocol. Caps are applied only at evaluation to the separately adapted NoPE/Polar checkpoints. Ten held-out examples per task and length; these are single-run results.", "tab:all_babi")
    return "% Generated by paper/generate_primary_tables.py; do not edit.\n" + loss + "\n\n" + reasoning + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tables = {"tab:endpoints": primary_endpoints(), "tab:replication_main": replication_endpoints()}
    appendix_tables = {
        "table:retrieval_breakdown": retrieval_breakdown("token_accuracy", "table:retrieval_breakdown"),
        "table:retrieval_exact_breakdown": retrieval_breakdown("exact_match", "table:retrieval_exact_breakdown"),
    }
    for edition, filename in (("iclr2027", "iclr2027_conference.tex"), ("arxiv", "atma_arxiv.tex")):
        reference_path = ROOT / "paper" / edition / "reference_results_tables.tex"
        generated = reference_tables()
        if args.check:
            if not reference_path.exists() or reference_path.read_text(encoding="utf-8") != generated:
                raise SystemExit(f"stale tables: {reference_path}")
        else:
            reference_path.write_text(generated, encoding="utf-8")
        for name, replacements in ((filename, tables), ("appendix.tex", appendix_tables)):
            path = ROOT / "paper" / edition / name
            original = path.read_text(encoding="utf-8")
            updated = original
            for label, replacement in replacements.items():
                updated = replace_table(updated, label, replacement)
            if args.check:
                if updated != original:
                    raise SystemExit(f"stale tables: {path}")
                print(f"verified {path.relative_to(ROOT)}")
            else:
                path.write_text(updated, encoding="utf-8")
                print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
