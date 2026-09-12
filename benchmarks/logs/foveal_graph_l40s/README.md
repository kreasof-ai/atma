# Corrected Foveal graph-decoder measurements

The [serving report](../../../foveal_cpt/SERVING_L40S.md) replaces the earlier eager-decoder performance interpretation. `summary.json` retains medians, repetition ranges, dense baselines, numerical summaries and source checksums. `protocol.json` identifies decoder commit `36373b7` and SHA-256 hashes of all three decoder source files.

The twelve serving JSONs contain one warmup and three measured fresh-state requests per model/length. All generated 130 tokens with 129 timed graph decode steps, and every nonlocal attention layer retained active routing. `dense_*` measures the existing full-attention source-checkpoint engines at 2K with the same token budget; their graph boundaries and scheduler differ from the Foveal engine. `validation_*` retains BF16 teacher-fed comparisons, full-forward numerical controls and free-running tokens. `fp32_*` retains the unchanged strict tolerance results at 641- and 3,968-token prefixes, including an active 32-remote-page cap.

`regressions.log` records 124 passing tests. `eager_profile.log` and its provenance file explain the earlier implementation bottleneck; profiler timings include instrumentation overhead. `setup_failures/` retains initial helper-import failures, which produced no measurements. The corrected helper runs are recorded in `rechecks.console.log`; the canonical summaries use their successful reports.

From the repository root, with the recorded CUDA environment and pinned checkpoint snapshot:

```bash
OMP_NUM_THREADS=4 HF_HUB_DISABLE_PROGRESS_BARS=1 python -m benchmarks.audit_foveal_serving \
  --model /path/to/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/polar/kl/cpt \
  --context_tokens 2048 --decode_tokens 130 --warmup_samples 1 --samples 3 \
  --out /tmp/foveal-graph-polar-kl-2k.json

OMP_NUM_THREADS=4 python -m benchmarks.verify_foveal_cache \
  --model /path/to/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/polar/kl/cpt \
  --cuda_graph --float32_oracle --lengths 641 3968 \
  --decode_steps 66 --compare_every 16 --out /tmp/foveal-graph-fp32.json
```

The two helper scripts preserve the exact dense-baseline and BF16 control procedures. Use new output paths. The first request includes graph preparation/capture; warm requests reuse graphs and persistent compiler caches. These experiments do not assert isolated cold-compilation costs, batching throughput, or 512K/1M capacity.
