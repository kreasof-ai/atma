# L40S claim-review evidence (2026-09-12)

[claim_evidence.json](claim_evidence.json) summarizes the completed comparisons for all twelve pinned CPT checkpoints. [AUDIT_FOLLOWUP.md](../../../foveal_cpt/AUDIT_FOLLOWUP.md) states what they do and do not support. No serving, cold-compilation, or 512K/1M capacity measurement was collected.

[l40s_cache_evidence.tar.gz](l40s_cache_evidence.tar.gz) preserves the raw JSON reports, console logs, environment, checkpoint hashes, and diagnostic controls without adding hundreds of thousands of generated text lines to Git. Its checksum is in the summary. Extract it into a fresh directory:

```bash
mkdir -p /tmp/foveal-l40s-evidence
tar -xzf benchmarks/logs/foveal_verified/l40s_cache_evidence.tar.gz -C /tmp/foveal-l40s-evidence
```

The summary's `sources` select 120 completed reports: for each checkpoint, six BF16 teacher-fed cases, three FP32 cases, and one report containing two free-running continuations. Counts exclude the short teacher-fed prechecks inside the free-running reports. The archive also retains exploratory/superseded runs and the twelve unsupported `fp32_routing` attempts. They are not silently counted as successful comparisons.

The original `numerical_protocol.json` records the broader planned audit, including an empirical numerical envelope; it is not a claim of completion or a formal error bound. The archived `cache_summary.json` remains incomplete because the planned 8K FP32 reference exceeds its 4,096-token limit. Serving and compile watcher logs contain no measurements. The broad runner's default FP32 routing case has this known limit; use the explicit completed case list below to reproduce only this evidence.

Each report records its command/settings, checkpoint path, GPU, source commit and decoder hash. Some early reports ran with uncommitted decoder corrections; the hash, rather than their earlier HEAD alone, identifies that decoder. `environment.json` records the dependency stack and pinned causal-conv kernel; exact equivalence to the historical benchmark environment is not claimed. Offline Hub mode can select a different convolution fallback, so the archived offline pilot is excluded from the canonical counts.

To repeat the completed numerical cases with the recorded dependencies and pinned local checkpoint snapshot:

```bash
python -m benchmarks.run_foveal_audit \
  --checkpoint_root /path/to/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3 \
  --out_dir /tmp/foveal-new-audit --phase cache \
  --cases boundaries long continuation prose generated_pages routing \
          fp32 fp32_continuation fp32_generated_pages greedy
```

Strict verifier failures remain failures at `atol=rtol=1e-4`. The FP32 sequential-memory controls explain two small accumulation-order outliers; they do not rewrite raw results. The BF16 free-running divergence is retained, including controls that do not reproduce its token flip. Numerical diagnostics do not certify retrieval quality, serving performance, or the training-data protocol.
