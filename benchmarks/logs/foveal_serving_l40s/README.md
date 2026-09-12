# Focused Foveal serving measurements

See [SERVING_L40S.md](../../../foveal_cpt/SERVING_L40S.md) for the results and limitations. `protocol.json` fixes the selected checkpoints and request counts; `environment.json` records the stack; `summary.json` contains medians, ranges, matched overhead and source checksums. Each model/length JSON retains its exact command, loading/first-request costs and all three measured requests. Console logs preserve errors and completion status.

To reproduce a cell on the recorded CUDA stack with the pinned checkpoint snapshot:

```bash
OMP_NUM_THREADS=4 HF_HUB_DISABLE_PROGRESS_BARS=1 python -m benchmarks.audit_foveal_serving \
  --model /path/to/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3/polar/kl/cpt \
  --context_tokens 32768 --decode_tokens 130 --warmup_samples 1 --samples 3 \
  --out /tmp/foveal-polar-kl-32k-serving.json
```

Use a new output location. Hub offline mode can change the convolution backend. The first request is excluded from warm statistics; persistent compiler caches are reused. These timings retain the numerical limitations stated in the claim audit and do not assert exact BF16 equivalence.
