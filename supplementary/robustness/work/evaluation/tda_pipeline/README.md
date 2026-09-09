# Completed TDA 10B benchmarks

All nine pipeline stages completed. `progress.json` records the SHA-256 of each
completed stage artifact; these hashes were verified before committing this bundle.
`plan.json` records the exact commands, base checkpoint hash, dataset manifest hash,
and source file hashes used for the run. The recorded sources matched the submitted
implementation at artifact verification.

- `results/`: full downstream, synthetic/real retrieval, BPB, direct serving, and
  adapted BABILong structured logs.
- `benchmark_matrix.json` and `.csv`: 537 aggregated rows from those result logs.
- `gate/`: the separate one-example 256K BABILong feasibility gate.
- `console/`: per-stage console output, including checkpoint verification and adaptation.
- `babilong_checkpoint/`: adapted checkpoint configuration, tokenizer, fine-tuning
  manifest, and training summary. The large `weights.pt` remains local and is excluded
  from Git; its SHA-256 is in `progress.json`. The base weights also remain local;
  their SHA-256 is in `plan.json`.

Serving results use `direct-full-recompute-v1`, not cached decoding. Keep them
separate from the existing paged-serving results. BABILong uses a separate adapted
checkpoint; it does not modify the original 10B weights. The full benchmark commands
and interpretation constraints are in `../../../TDA_BENCHMARKS.md`.

The [TDA benchmark interpretation](../../../TDA_BENCHMARKS.md#completed-benchmark-interpretation-2026-09-09)
summarizes these results and the subsequent gamma inspection. Both the base and
BABILong-adapted checkpoints have a maximum zero-input half-life of 36.74 tokens
(block 6, head 7); none of their 64 heads exceeds 256 tokens at zero input. A second
full gamma-capped suite is deferred, with token-dependent gamma explicitly left
unresolved. The scan is preserved in [the gamma diagnostic bundle](../tda_gamma256/parameters/gamma_parameters.json).
