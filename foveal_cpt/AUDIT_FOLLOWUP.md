# Foveal correctness audit: changes and remaining validation

Updated 2026-09-12. This follow-up covers the stored `benchmarks/logs/foveal_cpt` evaluation snapshot and `baseline_inference/foveal_engine.py`. It does not certify the training-data protocol or replace other research gates in [README.md](README.md).

## Completed locally

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

## Local environment and test evidence

This Windows machine reports an **AMD Radeon RX 6700 XT**, with no NVIDIA CUDA runtime. The isolated test environment uses Python 3.12 and **PyTorch 2.14.0+cpu**. The trained CPT weights are not present locally; the recorded checkpoints are about 3 GB each. CUDA Triton and FLA inference paths cannot be exercised here.

Run the CPU suite from the repository root:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = (Get-Location).Path
uv run --python 3.12 --with torch --with pytest --with transformers --with kernels --with datasets python -m pytest tests/test_foveal_engine.py tests/test_foveal_aggregation.py tests/test_foveal_eval.py tests/test_benchmark_pipeline.py -q
```

Local result on 2026-09-12: **78 passed** in 21.74 seconds. The only warning was the expected Hugging Face offline-mode trust-check warning. All 192 retrieval table values were also checked against the corrected matrix, and regenerating the matrix verified all 72 selected source checksums.

The additional `tests/test_baseline_inference.py` run reached five dependency failures: two Softmax layout tests require Triton and three Raven layout tests require FLA. Its CUDA kernel test was skipped. These are still required checks on the CUDA machine because the shared weight loader changed; they have not been marked as passing.

## Remaining GPU and checkpoint audit

Use the original Linux/CUDA environment with the same PyTorch, Triton, causal-conv and FLA versions used for the L40S runs. Record the GPU, dependency versions, repository commit, checkpoint revision and weights path with each result. The original snapshot revision recorded in the logs is `e4cf2558793646c26d9e5a6c6eeadb4e1011cad3` of `ChavyvAkvar/atma-foveal-cpt-all`.

1. **Run the local regressions plus CUDA dependencies.** Run the four CPU suites above, `tests/test_baseline_inference.py`, and the project's relevant sparse-attention/FLA tests on the GPU environment. The new GPU prefill-state extraction follows the existing inference FLA `[K,V]` convention, but is not locally GPU-verified.

2. **Verify every real checkpoint before timing it.** [verify_foveal_cache.py](../benchmarks/verify_foveal_cache.py) compares prefill and cached-step logits with full recomputation on the same checkpoint and identical input tokens. Default cases include prompts of 1, 2, 63, 64, 65, 511, 512, 513, 639, 640 and 641 tokens, followed by 66 teacher-fed decode steps. This exercises short histories, partial pages, new pages and remote eligibility. It records maximum logit errors, tolerance results and greedy-token agreement for every position, and exits nonzero on failure. Repeat on representative natural-language and retrieval token streams if extending the verifier; the built-in varied records are a diagnostic workload, not a quality benchmark.

```bash
# Run from the repository root on Linux; replace this with the pinned snapshot directory.
CPT_ROOT=/path/to/atma-foveal-cpt-all/snapshots/e4cf2558793646c26d9e5a6c6eeadb4e1011cad3
python -m benchmarks.verify_foveal_cache \
  --model "$CPT_ROOT/polar/lm_output_kl/cpt" --device cuda \
  --decode_steps 66 --out benchmarks/logs/foveal_verified/parity_polar_lm_output_kl.json
```

Repeat for `polar`, `nope`, `rope` crossed with `local`, `lm_output`, `kl`, `lm_output_kl`, and then with longer prefixes (for example `--lengths 2048 8192 32768 --decode_steps 2`). Also test longer continuations exceeding 512 generated tokens so generated pages become remote candidates. Real-checkpoint free-running greedy generation should be compared after teacher-fed parity passes.

The default `atol=rtol=1e-4` is a strict diagnostic starting point, **not an established BF16 acceptance threshold**. GPU kernels and BF16 matmul shapes can introduce legitimate differences. If it fails, quantify those against a matched numerical control and inspect routing/state differences before setting an explicit tolerance; record the rationale and all greedy disagreements. Do not loosen tolerances solely to obtain PASS. CPU float32 parity does not justify a blanket tolerance for all CUDA backends.

3. **Collect actual Foveal serving measurements after parity passes.** Use a fresh output directory so the pipeline cannot resume the historical ordinary-engine results. Run on an otherwise idle L40S for comparability. The engine is currently serial and uses full-prefix KV storage with tensor concatenation; there is no CUDA-graph, constant-time cache-append, batching-throughput or flat-latency guarantee.

```bash
python -m benchmarks.run --benchmark serving \
  --model "$CPT_ROOT/polar/lm_output_kl/cpt" \
  --lengths 2k 4k 8k 16k 32k 64k 128k 256k \
  --decode_tokens 130 --serving_samples 3 --serving_warmup_samples 1 \
  --max_num_seqs 1 --strict \
  --out benchmarks/logs/foveal_verified/serving_polar_lm_output_kl.log
```

Repeat for all twelve variants. Check `backend=FovealLLM`, `protocol=exact-token-prefill-v2`, `ignore_eos=true`, and measured decode-token counts. With 130 output tokens, each request performs 129 timed cached decode steps. Report cold-start/compile cost separately from warm timings, per-length distributions across independent repetitions, prefill time, decode time and both allocated/reserved peak memory. Differences from the old 32-token protocol must be stated. Run a second matched 32-token sweep if comparing those budgets specifically; never relabel the historical index-disabled numbers as new Foveal results.

4. **Test 512K/1M capacity only as a new experiment.** After shorter parity and serving checks, repeat a bounded sweep at `--lengths 512k 1m` on the 48 GB target. Record OOM outcomes, full-prefix cache size and allocated/reserved peaks. The former docstring claim that these lengths fit in 48 GB has been removed; it is unsupported until measured. Full-forward prefill, index scores, memory-state extraction and KV append traffic can dominate these lengths even though attention gathers a bounded active set.

None of steps 1–4 is marked complete for the real CUDA checkpoints by this local work.

## Rebuilding the audited historical report

```bash
python -m benchmarks.aggregate --log_dir benchmarks/logs/foveal_cpt
```

The manifest makes this directory a frozen historical snapshot. A changed selected log fails its checksum; duplicate selected result cells fail aggregation. New logs in this directory are not automatically included while the manifest exists. Place new experiments in a new directory and select their intended repetitions explicitly before making summary tables. Retain repetition-level data when reporting variability rather than averaging conflicting experiments by accident.

Raw retrieval/base/longdoc/BABILong quality measurements and gamma parameter records were not rerun or altered. The corrected aggregate, historical backend annotations and documentation can be used now. Publish new Foveal cached-serving conclusions only after the remaining verification above.
