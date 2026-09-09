# Completed TDA 10B checkpoint: benchmark handoff

Run every group sequentially on GPU 0 with:

```bash
python -m supplementary.robustness.benchmark_tda --gpu 0
```

Add `--dry-run` to print the commands. The runner writes logs, a separate BABILong
checkpoint, progress records, and aggregate JSON/CSV under
`supplementary/robustness/work/evaluation/tda_pipeline/`. Repeating the command
skips verified completed stages. It refuses to mix changed code/checkpoints into
an existing run and preserves partial fine-tuning checkpoints rather than overwriting
them. Failed stages stop the pipeline; the 256K BABILong gate must complete without
OOM before the full adapted sweep. Systems measurements use full-prefix recomputation.

Run from the repository root with the pinned TDA environment. These commands use
`checkpoints/supplementary_robustness/scaled_tda_hybrid` as an immutable base checkpoint.
No training configs or pilot artifacts are modified. Each command is a fresh process,
so training CUDA graphs and optimizer allocations cannot carry into evaluation.

The generic `benchmarks.run_pipeline` and `run_babilong_pipeline` still target the
original five Hub models. Use the individual entry points below for the local TDA
checkpoint. Each writes its own structured result log. These individual commands do
not skip completed jobs automatically; inspect existing logs before rerunning them.

## Verification

```bash
python -m external_baselines.verify_checkpoint checkpoints/supplementary_robustness/scaled_tda_hybrid
```

The scorer verifies configured dependency commits and clean tracked dependency sources,
then loads the external model with checkpoint-key mismatch checks. Downstream scoring,
retrieval, BPB, and BABILong use that same checkpoint-exact direct model. Dataset
revisions below match the original benchmark artifacts rather than resolving new main
revisions. Benchmark randomness and sample budgets retain the established defaults.

## Downstream quality

All eight existing tasks, complete evaluation splits, 2K scoring context, batch size 8:

```bash
python -m benchmarks.run --benchmark base \
  --model checkpoints/supplementary_robustness/scaled_tda_hybrid \
  --batch_size 8 --scoring_max_length 2048 \
  --dataset_revisions benchmarks/logs/atma_10b/checkpoint_manifest.json \
  --out supplementary/robustness/work/evaluation/tda/base.log --strict
```

## Retrieval

Five-token teacher-forced passkey and NIAH, 50 paired samples at each of three depths,
with synthetic and FinePDFs distractors. The 256K cells are attempted and OOMs recorded;
contexts must not be silently shortened to the training run's 128K intrinsic limit.

```bash
python -m benchmarks.run --benchmark retrieval \
  --model checkpoints/supplementary_robustness/scaled_tda_hybrid \
  --tasks passkey niah --lengths 2k 4k 8k 16k 32k 64k 128k 256k \
  --depths 0.1 0.5 0.9 --samples 50 --seed 1234 --retrieval_value_tokens 5 \
  --out supplementary/robustness/work/evaluation/tda/retrieval_synthetic.log --strict

python -m benchmarks.run --benchmark retrieval \
  --model checkpoints/supplementary_robustness/scaled_tda_hybrid \
  --tasks passkey niah --lengths 2k 4k 8k 16k 32k 64k 128k 256k \
  --depths 0.1 0.5 0.9 --samples 50 --seed 1234 --retrieval_value_tokens 5 \
  --haystack codelion/finepdfs-1B \
  --haystack_revision 6fbca6dd53e3e794f607cfe1f5a91c07e17b01ab \
  --out supplementary/robustness/work/evaluation/tda/retrieval_real.log --strict
```

## Fixed-target BPB

Same three datasets and 256-token targets; eight requested documents per dataset,
with actual qualifying counts reported by the harness:

```bash
python -m benchmarks.run --benchmark longdoc \
  --model checkpoints/supplementary_robustness/scaled_tda_hybrid \
  --datasets finepdfs pg19 proof_pile --lengths 2k 4k 8k 16k 32k 64k 128k 256k \
  --target_tokens 256 --num_docs 8 --max_scan 100000 \
  --dataset_revisions benchmarks/logs/atma_10b/checkpoint_manifest.json \
  --out supplementary/robustness/work/evaluation/tda/bpb.log --strict
```

## BABILong adaptation and evaluation

Create a separate adapted checkpoint with the existing recipe: qa1–10; 0K/1K/2K
training; rows 0–79 train and 80–89 validation; three epochs; answer/EOS-only loss;
validation-best checkpoint selection; seed 1234. The default optimizer settings are
unchanged. Use the same `builtin-v1` prompt environment as the published runs; do not
install the optional `babilong` prompt package between adaptation and evaluation.

```bash
python -m benchmarks.finetune_babilong \
  --model checkpoints/supplementary_robustness/scaled_tda_hybrid \
  --output_dir checkpoints/supplementary_robustness/tda_babilong_2k_ft \
  --dataset_revision ee0d588794c7ac098062ee0d247c733d62e94fe2 \
  --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10 \
  --train_lengths 0k 1k 2k --seq_len 2048 \
  --train_start 0 --train_end 80 --val_start 80 --val_end 90 \
  --epochs 3 --micro_batch_size 1 --grad_accum_steps 8 --seed 1234

python -m benchmarks.run --benchmark babilong \
  --model checkpoints/supplementary_robustness/tda_babilong_2k_ft \
  --tasks qa1 --lengths 256k --row_start 90 --row_end 100 --samples 1 \
  --babilong_backend direct --max_tokens 16 \
  --out supplementary/robustness/work/setup/tda/babilong_gate.log --strict
```

Inspect the gate log for a completed non-OOM response before the full sweep. A process
exit code alone is insufficient because the harness records OOM cells in its results.
The evaluator inherits the adapted checkpoint's pinned dataset revision and validates
its prompt/held-out protocol.

```bash
python -m benchmarks.run --benchmark babilong \
  --model checkpoints/supplementary_robustness/tda_babilong_2k_ft \
  --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10 \
  --lengths 0k 1k 2k 4k 8k 16k 32k 64k 128k 256k \
  --row_start 90 --row_end 100 --samples 10 --babilong_backend direct --max_tokens 16 \
  --out supplementary/robustness/work/evaluation/tda/babilong.log --strict
```

## Inference systems scaling: direct full-prefix backend

TDA has no cached decoding backend in this repository. `--serving_backend direct`
measures the exact training forward, recomputing the entire growing prefix for every
new token. It must be reported separately from paged/cached serving results. This is
an executable baseline measurement, not an optimized TDA serving implementation.

```bash
python -m benchmarks.run --benchmark serving --serving_backend direct \
  --model checkpoints/supplementary_robustness/scaled_tda_hybrid \
  --lengths 2k 4k 8k 16k 32k 64k 128k \
  --decode_tokens 32 --serving_samples 1 --max_num_seqs 1 \
  --out supplementary/robustness/work/evaluation/tda/serving_direct.log --strict
```

This matches the original systems context grid. Add `256k` for a separately identified
extension. The backend warms every exact shape with one complete sequence, then times
prompt processing plus first-token selection separately from the remaining 31 generated
tokens. EOS is ignored to enforce a fixed budget. GPU synchronization brackets timings;
checkpoint loading and tokenization are excluded. Peak allocated/reserved memory includes
model residency after warmup. Samples run sequentially at batch size one. Outputs include
TTFT, per-decode-token latency, throughput, raw sample timings, and OOM status. Long-context
full-prefix decoding may take substantially longer than the original cached backends.

The protocol is `direct-full-recompute-v1`; aggregated systems rows use
`suite=direct_full_recompute` to prevent mixing these results with cached decoding.

## Aggregate

```bash
python -m benchmarks.aggregate --log_dir supplementary/robustness/work/evaluation/tda
```

The BABILong gate is written under `work/setup/tda`, outside the final aggregation
input, because it is a one-example feasibility check.
