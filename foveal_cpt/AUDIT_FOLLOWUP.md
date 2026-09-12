# Foveal audit follow-up: what the evidence supports

Updated 2026-09-12 after checking the real checkpoints on this machine's NVIDIA L40S.

**The adaptation claim is defensible as a limited empirical result:** in this sweep, KL-trained Polar and NoPE variants improve synthetic needle retrieval over their local CPT controls, with similar total retrieval-evaluation runtime. This is teacher-forced answer-token accuracy under the recorded flat-stream training protocol. It does not establish reliable free-running retrieval, general long-context reasoning gains, or Foveal serving speed/capacity.

## Claim assessment

| Claim | Assessment |
| --- | --- |
| KL-trained adaptation improves synthetic retrieval over local CPT | Supported for the recorded checkpoints and workloads. At 256K, Polar `kl` scores 81.0% and NoPE `lm_output_kl` 82.7% token accuracy; both local controls score 0.0%. These are token scores, not exact-answer or free-generation success rates. |
| Retrieval gains come at near-SWA runtime | Supported only for the matched full-forward retrieval evaluations in Section 6 of [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). End-to-end totals include work beyond attention; no isolated kernel or cached-decode speedup follows. |
| Adaptation generally improves long-context reasoning or real-text retrieval | Not established. Real-text distractors sharply reduce retrieval. At 256K BABILong, Polar local scores 40% versus 36–38% for its index variants; RoPE LM-output also scores 40%. |
| KL is universally necessary, or core rankings prove an architectural mechanism | Not established by one trained checkpoint per cell. KL cells also receive 20M-token calibration outside the 1B CPT budget. Core comparisons include differences inherited from source checkpoints; no multi-seed uncertainty estimate or isolated mechanism ablation is provided. |
| Historical ~450–480 token/s results measure Foveal serving | Unsupported: the historical loader dropped the Foveal index/route parameters and LM-output residual. No replacement serving, cold-compilation, or 512K/1M capacity measurements were collected in this follow-up. |
| This completes the prescribed scientific training protocol | No. The unresolved [data protocol gate](README.md#run-gates) concerns flat streams without document-boundary attention/memory resets. The results must be described under the actual protocol. |

## Completed L40S checks

All twelve CPT checkpoints came from `ChavyvAkvar/atma-foveal-cpt-all` at revision `e4cf2558793646c26d9e5a6c6eeadb4e1011cad3`. The machine used PyTorch 2.14.0+cu130, Triton 3.8.0 and FLA 0.5.2. Exact dependency, base-checkpoint and weight-hash records accompany the [evidence summary](../benchmarks/logs/foveal_verified/claim_evidence.json) and [compressed raw reports](../benchmarks/logs/foveal_verified/l40s_cache_evidence.tar.gz).

The GPU checks found and corrected three cached-inference numerical issues: convolution products now accumulate in FP32 before rounding, Polar decode scores accumulate in FP32, and CUDA memory decode uses FLA recurrent state/output conventions. These changes did not modify training, `FovealAttention`, or `DirectScorer`. The historical full-forward quality results therefore do not depend on the defective ordinary serving path, and these decoder fixes alone do not require rerunning that quality matrix.

- **Regression evidence:** 96 combined tests and 7 additional audit tests passed (103 distinct tests). Sparse Polar forward/backward, NoPE/RoPE CUDA backward, and FLA state-layout checks also completed successfully.
- **BF16 diagnostics:** 16,956 teacher-fed comparisons covered page/window boundaries, prefixes through 32K, and continuations through 642 tokens, including remote pages created during generation. There were 16,716 strict `atol=rtol=1e-4` failures and 48 greedy-token disagreements. This strict diagnostic is not an established BF16 acceptance threshold. Full-forward shape/reduction and recurrent-memory controls also show numerical differences; these do not erase the failures or establish universal equivalence.
- **FP32 diagnostics:** 660 sampled comparisons had identical greedy tokens and final route sets. Two Polar LM-output+KL comparisons failed the unchanged strict tolerance, with maximum logit errors 0.00022304 and 0.00021911. Both matched a separate sequential-memory full forward at that same tolerance. The raw failures remain recorded.
- **Free-running check:** 23 of 24 130-token continuations matched exactly. RoPE LM-output+KL with a 65-token prompt diverged at the 16th generated token. Its full-forward numerical controls did not reproduce the token flip. Exact BF16 greedy equivalence remains unresolved.

The twelve extra 8K FP32 routing attempts exceeded the eager reference's 4,096-token limit and produced no comparisons. They are excluded from the completed counts. The planned serving/capacity sweeps were stopped before measurement and are outside this claim review. The diagnostic runner and its conservative serving gate remain experimental; no gate result is a quality certification. See the [artifact notes](../benchmarks/logs/foveal_verified/README.md) for scope and reproduction.

## Earlier correctness and reporting fixes

- Corrected cached generation: preserve Titans prompt memory in the decoder's `[K,V]` layout; exclude right padding from state; rotate index keys before page pooling; keep unrotated RoPE memory Q/K; retain the current block's routing anchor; finish index pages during generation; zero-pad short convolution histories; enforce minimum remote pages; honor temperature, EOS and token budgets; use configured dimensions, page size and local window.
- `EvalModel` now dispatches Foveal checkpoints to `FovealLLM`. The ordinary weight loader rejects Foveal weights instead of silently discarding index/route parameters, even in non-strict mode.
- Added [CPU regression tests](../tests/test_foveal_engine.py), comparing actual cached logits with full recomputation across all twelve core/mode combinations. Small fixtures have two attention layers, active nonzero memory and index residuals, and cross several page boundaries into remote support. Additional cases use production page/window sizes and one production-width attention layer. Tests are float32 and do not establish GPU/BF16 parity.
- Made checkpoint-resolution unit tests self-contained and portable; they no longer require a remote base config or Unix-only temporary paths.
- Added an explicit [aggregation manifest](../benchmarks/logs/foveal_cpt/aggregation_manifest.json). It selects 72 complete runs, excludes 24 smoke runs and 13 superseded shorter runs, and checks LF-normalized SHA-256 source identities. Regenerated JSON/CSV contain **6,336 unique result rows**, replacing 7,146 mixed rows. Raw `.log` and `.console.log` files are unchanged.
- Corrected retrieval tables, including untouched values in the clamped comparison, from the selected full runs. NoPE-KL synthetic 2K token accuracy is **94.33%**, replacing the mixed-run **91.8%**.
- Relabeled historical serving rows as `ordinary_engine_index_disabled`. These measurements describe CPT backbone weights executed by ordinary sliding-window engines, not faithful sparse Foveal serving. Unknown Foveal serving backends are labeled `unverified_generation`; new Foveal runs identify `backend=FovealLLM` and `suite=foveal_cached_generation`.
- Added actual retrieval elapsed-time comparisons to [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Those same full-forward runs support retrieval gains at near-SWA evaluation runtime. Their elapsed time is not cached per-token decode latency.
- New serving protocol `exact-token-prefill-v2` records the backend, ignores EOS for a fixed token budget, and supports explicit unmeasured warmup requests. No new serving measurements have been collected here.

The archived quality runs remain usable under their recorded protocols. Fixing this separate decoder did not modify `FovealAttention`, training, or `DirectScorer`; no complete quality-suite rerun is indicated by these fixes alone. If subsequent GPU fixes touch the shared forward/scoring code, rerun the affected quality experiments.

## Status

The claim review is complete with the qualifications above; the broader serving audit is not complete. Preserve the observed retrieval result, distinguish token scoring from generation, and retain the training-protocol limitation. Any future serving or capacity claim needs its own successful validation and fresh measurements with `FovealLLM`.

The historical quality snapshot remains frozen. Rebuild its checksum-validated aggregation with:

```bash
python -m benchmarks.aggregate --log_dir benchmarks/logs/foveal_cpt
```

The earlier Windows CPU run passed 78 tests but could not exercise CUDA. Its missing CUDA dependencies are addressed by the L40S checks above. Raw historical quality logs and gamma records were not rerun or changed.
