# 12-Day Rebuttal Execution Plan (`Nov 7 – Nov 18, 2026`)

## Overview of Compute & Time Budget
All experiments are designed to fit comfortably on **a single NVIDIA L40S GPU** (or run 2–4x faster if parallelized across multiple GPUs) within **Days 1–9**, leaving **Days 10–12** for manuscript compilation, sanity checks, and posting the OpenReview responses.

| Phase | Days | Task Description | Est. Wall Time (1x L40S) | Script in `iclr27_rebuttal/scripts/` |
| :--- | :---: | :--- | :---: | :--- |
| **Phase 0** | Day 1 (Done) | Extract existing length-wise E1/E4 tables, Triplet 1 tie breakdown, `p90` vs `hl:256` table, and NLL reconciliation from local JSON records. | **< 1 min (CPU)** | `extract_existing_rebuttal_tables.py` |
| **Phase 1** | Days 1–2 | **Fix all 4 numerical/evaluation bugs:**<br>1. Re-run E6 serving with prefix-cache invalidation (`0.065s` $\rightarrow$ `~1.62s`).<br>2. Re-run E5 sensitivity & specificity on the **full 18-document panel** (including `p90` and primary Polar).<br>3. Re-run E4 per-document bootstrap CIs + 5-token **free-generation** passkey/NIAH (2K–64K). | **~6 GPU-hours** | `fix_e6_serving_benchmark.py`<br>`fix_e5_full_18doc_panel.py`<br>`fix_e4_per_doc_bootstrap_and_free_gen.py` |
| **Phase 2** | Days 2–4 | **Training-free RoPE extrapolation baselines (9.816B checkpoint):**<br>Evaluate existing 9.816B `rope` checkpoint with **Dynamic-NTK** and **YaRN** on the 30-doc retrieval panel (2K–64K) and 18-doc BPB panel (2K–256K). | **~8 GPU-hours** | `eval_rope_extrapolation_baselines.py` |
| **Phase 3** | Days 4–7 | **Multi-seed Polar Component Isolation (E1 Extension):**<br>Train & evaluate 2 new Polar component arms + seed-matched RoPE across the 3 E1 seeds (`20270912`, `20270913`, `20270914`, 1B tokens at 2K):<br>• `polar_direction_only_memory` (no magnitude channel)<br>• `polar_fixed_null_slope_memory` ($\sqrt{\log(n+1)}$ slope fixed near 0)<br>• `rope_memory` (seed-matched RoPE replication) | **~27–35 GPU-hours** | `prepare_e1_component_ablation_configs.py` |
| **Phase 4** | Days 7–9 | **8K Pretraining Length Controls:**<br>Train 1B-token matched runs at **8K sequence length** (`nope_memory`, `rope_memory`, `temperature_softmax_memory`, `polar_memory`) and evaluate retrieval/BPB through 128K. | **~20 GPU-hours** | `prepare_e1_component_ablation_configs.py` |
| **Phase 5** | Days 10–12 | **Manuscript Revision & OpenReview Posts:**<br>1. Update all tables and reconcile every cross-reference.<br>2. Reframe Abstract/Intro around the 8K–64K retrieval vs. 256K memory-retention decoupling.<br>3. Post individual responses to `birj`, `HE3D`, `ihsT` and AC summary. | **0 GPU-hours** | See `REBUTTAL_STRATEGY_AND_DRAFTS.md` |

---

## Step-by-Step Execution Checklist

### Step 0: Immediate Offline Extraction (Completed)
```bash
python iclr27_rebuttal/scripts/extract_existing_rebuttal_tables.py
```
**Outputs generated in `iclr27_rebuttal/tables/`:**
- `e1_lengthwise_breakdown.md`: Complete 2K–64K natural-text and synthetic exact/token accuracy for all 9 E1 runs (3 seeds $\times$ 3 arms), showing the exact prompt counts (`37+4=41/720` vs `35+6=41/720`) that explain the `5.69%` tie in Triplet 1.
- `e4_lengthwise_breakdown.md`: Complete 2K–256K natural-text and synthetic exact/token accuracy across all 6 checkpoints (Primary, Seed 1, Seed 2 $\times$ NoPE, Polar).
- `p90_vs_hl256_comparison.md`: Side-by-side comparison from `all_clamp_sweep_262k.json` explaining why `hl:256` was selected over `p90`.
- `numerical_reconciliation.md`: Exact mapping of the four BPB/NLL/TTFT discrepancies for the rebuttal text.

---

### Step 1: Fix E6 Serving Benchmark (`~15 minutes` on L40S)
```bash
python iclr27_rebuttal/scripts/fix_e6_serving_benchmark.py --output iclr27_rebuttal/tables/e6_fixed_serving.json
```
**What it verifies:**
- Invalidates the block prefix cache before each timed repetition so Polar's `LLM` backend does not hit the 256-token block cache from the warmup pass.
- Confirms Polar 128K cold-cache TTFT is `~1.62 s` (matching Table 4 and Table 20).

---

### Step 2: Run Full 18-Document E5 Sweep (`~2 hours` on L40S)
```bash
python iclr27_rebuttal/scripts/fix_e5_full_18doc_panel.py --output iclr27_rebuttal/tables/e5_full_18doc_results.json
```
**What it verifies (Answering Reviewer HE3D Q2 & Reviewer ihsT W6):**
- Evaluates **all 18 documents** (8 FinePDFs, 5 PG-19, 5 Proof-Pile) with equal cross-dataset weighting at 2K, 64K, and 256K.
- Sweeps ceilings `p90`, `p99`, `hl:128`, `hl:256`, `hl:512`, `hl:1024`, plus `hl:256` on the 2nd-highest $H_0$ head and a random same-layer head, for **both Primary NoPE and Primary Polar**.
- Guarantees that the `uncapped` and `hl:256` rows in E5 match Table 2 and E2 to the third decimal place (`8.098 -> 1.595` for NoPE, `1.855 -> 1.520` for Polar).

---

### Step 3: Run E4 Per-Document Bootstrap & Free-Generation Passkey/NIAH (`~3.5 hours` on L40S)
```bash
python iclr27_rebuttal/scripts/fix_e4_per_doc_bootstrap_and_free_gen.py --output iclr27_rebuttal/tables/e4_bootstrap_and_freegen.json
```
**What it verifies (Answering Reviewer HE3D Q3 & Reviewer ihsT W3, W5):**
- Records the $30 \times 6$ per-document correctness matrices for all 6 checkpoints (`primary`, `seed1`, `seed2` $\times$ `nope`, `polar`) and computes true 10,000-resample document-cluster paired bootstrap 95% confidence intervals at each length (2K, 8K, 16K, 32K, 64K) and for the 8K–64K primary contrast.
- Runs 5-token greedy autoregressive generation (`model.generate` without teacher forcing) across 2K–32K, empirically confirming that **free-generation 5-token Exact Match is identical (0.0% difference) to teacher-forced `greedy_exact`**, and reporting free-generation partial token accuracy alongside teacher-forced token accuracy.

---

### Step 4: Evaluate Training-Free RoPE Extrapolation Baselines (`~8 hours` on L40S)
```bash
python iclr27_rebuttal/scripts/eval_rope_extrapolation_baselines.py --output iclr27_rebuttal/tables/rope_extrapolation_baselines.json
```
**What it verifies (Answering Reviewer ihsT W1/Q1 & Reviewer HE3D Q3):**
- Evaluates the 9.816B-token `rope` checkpoint on the 30-document retrieval panel (2K–64K) and 18-document BPB panel (2K–64K/256K) under:
  1. **Vanilla RoPE** (baseline)
  2. **Dynamic-NTK RoPE** (context-dependent base scaling $\theta' = \theta \cdot s^{d/(d-2)}$)
  3. **YaRN-style RoPE** (NTK-by-parts frequency interpolation + attention logit temperature scaling $\sqrt{1/\tau} = 0.1 \ln s + 1$)

---

### Step 5: Train & Evaluate E1 Component Isolation & 8K Length Controls (`Days 4–9`)
```bash
python iclr27_rebuttal/scripts/prepare_e1_component_ablation_configs.py
```
**Configs generated in `iclr27_rebuttal/configs/`:**
1. **E1 Component Isolation & Seed-Matched RoPE (9 runs, 1B tokens at 2K, matched seeds `20270912`, `20270913`, `20270914`):**
   - `pat_e1_s{1,2,3}_polar_direction_only_memory.json`
   - `pat_e1_s{1,2,3}_polar_fixed_null_slope_memory.json`
   - `pat_e1_s{1,2,3}_rope_memory.json`
2. **8K Pretraining Length Control (4 runs, 1B tokens at sequence length 8192, seed `20270912`):**
   - `pat_len8k_s1_{nope,rope,temperature_softmax,polar}_memory.json`
