# ICLR 2027 Rebuttal War Room (`Submission12224`)

## 1. Current Standing & Target Score Math

| Reviewer | Rating (1–4) | Confidence (1–4) | Presentation (1–3) | Informativeness (1–3) | Soundness (1–3) | Primary Stance & Movability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Reviewer birj** | **2** (Weak Reject) | 2 | 2 | 3 | 2 | **High movability (2 $\rightarrow$ 3).** Appreciates the controlled attribution (`+22.18` pts), memory factorial (`+47.8` pts), and honest replication reporting. Main concerns are the unstable 256K headline in the abstract, single scale/length, and short-context/decode cost framing. |
| **Reviewer HE3D** | **1** (Clear Reject) | 3 | 1 | 1 | 1 | **Medium movability (1 $\rightarrow$ 2 or 3).** Praises the 20/20 factorial memory effect and rare honesty on negative results, but gave `Soundness: 1` due to **three concrete numerical discrepancies** (`0.065s` vs `1.619s`, `1.855` vs `1.825`, `10.77` vs `12.003`), E1 not separating direction vs. null, E5 using 6 docs instead of 18, and missing free-gen/YaRN baselines. |
| **Reviewer ihsT** | **1** (Clear Reject) | 3 | 2 | 2 | 1 | **Medium-Low movability (1 $\rightarrow$ 2 or 3).** Praises the factorial study, NoPE single-head failure discovery, and honest negative results, but gave `Soundness: 1` due to the **exact same numerical discrepancies**, Table 7 (`Fixed null slope` 91.25% > `Full Polar` 77.50%) conflicting with E1 claims, missing YaRN/ReRoPE baselines, capped NoPE (`1.506/1.490`) beating capped Polar (`1.644/1.762`) at 256K BPB, and missing length-wise tables/CIs in E4. |
| **Current Mean** | **1.33** | **2.67** | **1.67** | **2.00** | **1.33** | Bottom ~10% of ICLR 2027 submissions (mean across conference: `2.06`, top 25% cutoff: `2.33`, top 10% cutoff: `2.67`). |

### What We Need to Flip the Decision
To move into the acceptance conversation (`Mean >= 2.33–2.67`), we must:
1. **Eliminate every `Soundness: 1` objection** by fixing the three script/table bugs (`0.065s`, `1.855`, `0.0`-width bootstrap CI) and reconciling every table in the manuscript.
2. **Directly answer every factual question** with zero-training re-evaluations on existing checkpoints (full 18-doc E5 sweep, free-generation passkey verification, training-free RoPE+YaRN / Dynamic-NTK / ReRoPE baselines on the 9.8B RoPE checkpoint).
3. **Run targeted 1B-token matched experiments (~45–50 GPU-hours total on 1x L40S)** to isolate direction vs. null slope across the 3 E1 seeds and test an 8K pretraining length control.
4. **Reframe the paper's headline thesis** around the decoupling our data actually proves:
   - *Long-context likelihood stability (256K BPB)* is governed by **recurrent retention-gate margins** (where capping the single outlier head in NoPE restores 256K BPB to `1.49–1.60`, matching or beating Polar).
   - *Medium-to-long-range natural-text associative retrieval (8K–64K)* is governed by **attention selectivity** (where retention capping cannot rescue NoPE from `0.0%` beyond 8K, while Polar delivers a reproducible **`+37.18` percentage-point** exact-retrieval advantage across 30 independent documents and 3 seed pairs).

---

## 2. Directory Contents

- [`REVIEW_AUDIT_AND_ROOT_CAUSES.md`](./REVIEW_AUDIT_AND_ROOT_CAUSES.md): Complete forensic audit of all reviewer criticisms and the exact code/data bugs discovered in `supplementary/pat_feedback/`.
- [`EXECUTION_PLAN_12_DAYS.md`](./EXECUTION_PLAN_12_DAYS.md): Day-by-day 12-day execution schedule, GPU budget, and verification checklist.
- [`REBUTTAL_STRATEGY_AND_DRAFTS.md`](./REBUTTAL_STRATEGY_AND_DRAFTS.md): Rebuttal response templates for Reviewers `birj`, `HE3D`, and `ihsT`, plus the revised Abstract and Section 5.2/6.4 text.
- [`tables/`](./tables/): Pre-extracted Markdown and JSON tables generated from existing evaluation records (length-wise E1 and E4 breakdowns, Triplet 1 tie diagnosis, `p90` vs `hl:256` comparison, and NLL/BPB reconciliation).
- [`configs/`](./configs/): 13 generated JSON configs for the 3-seed E1 component ablations (`polar_direction_only_memory`, `polar_fixed_null_slope_memory`), seed-matched `rope_memory`, and 8K training-length controls.
- [`scripts/`](./scripts/): Ready-to-run Python scripts for all Day 1–9 experiments:
  - `extract_existing_rebuttal_tables.py`: Offline extractor that builds the missing length-wise tables and Triplet 1 diagnosis from existing JSON artifacts.
  - `fix_e6_serving_benchmark.py`: Fixes the 256-token prefix-cache hit bug in `run_e6_serving.py` that caused Polar's `0.065 s` 128K TTFT.
  - `fix_e5_full_18doc_panel.py`: Upgrades E5 to run on the full 18-document panel (with cross-dataset equal weighting matching Table 2) for both primary NoPE and primary Polar, including `p90`, `p99`, `hl:128..1024`, and alternative-head controls.
  - `fix_e4_per_doc_bootstrap_and_free_gen.py`: Fixes the degenerate bootstrap CI bug in `run_e4_experiment.py` and runs 5-token autoregressive greedy free generation (`model.generate`) alongside teacher forcing across 2K–64K.
  - `eval_rope_extrapolation_baselines.py`: Evaluates training-free RoPE extrapolation baselines (**Dynamic-NTK**, **YaRN**, **ReRoPE**) on the 9.816B RoPE checkpoint.
  - `prepare_e1_component_ablation_configs.py`: Generates the 3-seed E1 component isolation configs (`polar_direction_only_memory`, `polar_fixed_null_slope_memory`) and 8K training-length control configs.
