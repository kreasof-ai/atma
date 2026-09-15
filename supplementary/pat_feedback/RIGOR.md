# Rigor and acceptance standard

These requirements define valid evidence. They do not require a favorable score.
The current plan does not certify that existing runners already enforce them.

## 1. Freeze the comparison before production outcomes

Commit the protocol, resolved configs, evaluation manifests, primary contrasts,
selection rules, numerical tolerances, and analysis code before opening production
results. Record their hashes in a launch record. Prespecified seeds are not swapped
after a result is visible. Existing PAT, primary, replication, and Foveal results
informed this follow-up; say so rather than calling them prospectively registered.

An amendment records date, reason, affected runs, prior outcomes already visible,
resource impact, and the new protocol hash. Preserve the old version. A change
after viewing outcomes is identified in the paper, and any affected confirmatory
comparison becomes exploratory unless rerun under an appropriate new design.
An amendment can fix a real defect; it cannot erase a null result.

## 2. Matching means matching what executes

- E1 varies the attention operator only. Keep backbone, memory, preprocessing,
  tokenizer, training targets, microbatch, accumulation, device class, precision,
  optimizer groups, schedule, and checkpoint-selection rule fixed.
- Compare shared tensor names, shapes, values, and hashes **after** constructor
  initialization and all zero-initialization passes. Store a common-parameter map
  for differently named modules. Verify common gradients in the t=1 control test.
- Save exact parameter counts and the extra scalar/vector parameters per arm.
  Do not conceal meaningful compute or parameter changes with rounded "378M" labels.
- The current [RNG helper](../../train/reproducibility.py) explicitly says
  `data_seed` does not change shard order. E1 uses a common fixed token stream and
  three initialization seeds. Do not claim independent data-order repetitions.
- Inspect the actual batch tensors and shard positions for paired jobs. Logging
  equal config values does not prove equal consumed tokens. Include first-batch
  hashes, periodic batch hashes, and a final token-stream digest.
- Fixed batch 524,288 / (microbatch 4 x context 2,048) = 64 accumulation steps.
  If a device cannot fit that recipe, stop the family; do not shrink only one arm.
- Reusing existing source data or checkpoints requires exact hashes and compatible
  protocols. An older 1B seed is context, not one of E1's nine fresh matched runs.

The archived [component runner](../../ablation/train.py) zeros parameters whose
names contain `proj` and partitions parameters between Muon and AdamW. The new
implementation must resolve these behaviors explicitly before shared-init checks.
Do not rely on a vague "same optimizer" statement.

## 3. Immutable provenance and execution identity

Every run records code commit, tracked/untracked code state, config SHA-256,
Python/PyTorch/CUDA/Triton versions, dependency source SHAs, GPU model and memory,
driver, precision, compiler options, and whether deterministic kernels are available.
Use full source revisions and SHA-256 file hashes, not mutable `main`, timestamps,
or twelve-character prefixes as the sole identity. Keep a dirty patch artifact when
needed for development; production runs use a frozen, reconstructable code state.

Existing [runtime pins](../robustness/dependencies.json) are the candidate baseline,
not proof that the new temperature/sparse paths work. Run preflight on the actual
host. Do not silently upgrade a library, use an eager fallback, replace a fused
memory backend, or change device class within a matched comparison. If a necessary
change affects an operator, repeat all affected comparisons as a new version.

A model-condition fingerprint includes weights/config hashes, code/operator
version, tokenizer, fixture, dtype/backend, route settings, intervention target and
ceiling, task, length, document/case ID, target span and scoring protocol. Caches
reuse a result only on this complete identity. Prevent double counting when E2/E4
share the original uncapped cases.

For Foveal evaluation, verify that remote page selection, LM index output, final
routing schedule, and memory branch execute as trained. A loader accepting the
checkpoint is insufficient. A local-window serving approximation is a different
operator and cannot substantiate sparse-model quality or throughput claims.

## 4. Numerical acceptance

E0 first chooses and documents the intended finite-precision operator. The oracle
must implement the explicit equations and floors independently of kernel code.
Passing parity against another implementation with the same error is insufficient.

Starting tolerances below are fixed before production scores; any necessary
revision uses synthetic fixtures and a numerical error analysis, with a recorded
amendment. They are test-design choices, not claims that current kernels pass.

| Check | Required target |
|---|---|
| FP64 equation/reference values | atol 1e-10, rtol 1e-8 on scaled, non-overflowing fixtures |
| FP64 gradients away from floor boundaries | analytic/autograd and finite-difference agreement; normalized maximum error <= 1e-6 |
| FP32 reduction outputs | atol 1e-5, rtol 1e-4 |
| FP32 backward | normalized maximum gradient error <= 1e-3 |
| BF16 forward and backward | atol 5e-2, rtol/normalized maximum error <= 5e-2, with FP32 checks also passing |

Normalized maximum gradient error is max(abs(test-reference)) divided by
max(max(abs(reference)), 1e-6). Near-zero gradients also require max absolute error
<= 1e-6 in FP64, 1e-5 in FP32, and 1e-3 in BF16. Test and document the chosen
subgradient at exact floor boundaries separately from finite-difference checks.
Apply tolerances to each output/gradient tensor and test case, not just a global
mean that can hide an isolated failure. Record maximum and percentile errors.

Required properties include causal-mask correctness; temperature t=1 parity;
finite outputs/gradients; direction norm <= 1 within numeric tolerance; bounded
magnitude within dtype tolerance; correct uniform-weight participation ratio;
GQA/head mapping; and the selected all-null/padding convention. Deliberate
counterexamples near epsilon must pass the specified convention, not merely a
loose mixed-precision threshold. Avoid overflow during max-rescaling when the null
logit exceeds the real-key maximum.

Checkpoint impact reports include per-case loss/logit differences, floor activation
frequencies, and changed argmax/exact outcomes. Aggregate smallness is not a license
to alter existing results silently. No benchmark effect has been certified yet.

## 5. Real sampling units and statistical interpretation

Save per-document/per-trial measurements. Each real-text document is one cluster;
all its tasks, depths, target placements, and nested lengths remain together during
resampling. Use 10,000 paired bootstrap resamples with analysis seed 20270915 and
95% percentile intervals for document-conditioned effects. Recompute ratio metrics
from summed NLL and byte counts within each resample, not the mean of document BPBs.
For the three-corpus BPB mean, resample within each corpus and preserve equal corpus
weights.

Report per-initialization E1 effects, their mean/range, and document-conditioned
uncertainty separately. Thirty documents do not turn three training initializations
into thirty independent model replications. Do not treat individual digit tokens,
depths, or the 120 factorial settings as independent training seeds. Do not claim
population-level training-seed significance from a document bootstrap.

For synthetic retrieval, use paired case/target resampling, explicitly conditional
on the filler construction. Report this uncertainty separately from real-document
uncertainty. If every observed success count is zero, disclose the denominator and
bootstrap degeneracy; a [0,0] bootstrap interval is not proof of a zero population
success rate. Missing/failed cells are not zero accuracy.

The E1 primary contrast and E4 primary endpoint are fixed in the protocol. Other
corpora, lengths, cap ceilings, and metrics are descriptive secondary analyses.
Do not search them for a favorable p-value. No significance claim is required for
a complete experiment. Small differences or inconsistent seeds support a limited
or unresolved conclusion. Failure to reject a difference does not prove equivalence.

## 6. Gamma interpretation and intervention controls

Distinguish learned bias, fixed config bias, total zero-input logit, H0, actual
activation-conditioned logit, and observed gate/state/readout behavior. Use zero-based
transformer block and memory-head indices throughout. The head is a memory/query
head, not a KV-head index. Verify cap binding after compilation and wrapper loading.

H0 is a diagnostic operating point. It is not an empirical average runtime gate,
an effective memory lifetime of the full delta recurrence, or a proof of instability.
A long H0 with mild loss degradation challenges its sufficiency as a predictor;
it does not prove universal stability. The recurrence's theoretical retention margin
is sufficient, not necessary for finite-corpus empirical performance.

Keep likelihood, retrieval, and reasoning distinct. A cap that improves one can
harm another. A small downstream task mean shift does not establish preservation
of every task. A persistent parameter outlier does not establish unchanged memory
dynamics: Foveal CPT trains the backbone and can change W_gamma, activations,
writing, readout, and surrounding attention.

The [Foveal data protocol note](../../foveal_cpt/README.md) records packed streams
without document-boundary attention/memory resets. Reconcile that deviation in the
provenance record and describe the actual training regimen. The existing adapted
checkpoint can be studied as that regimen; it must not be represented as evidence
from coherent 32K-document training. This plan does not silently retrain it or
attribute the entire adaptation effect to sparse attention alone.

## 7. Failure handling, completion, and retention

Separate `process_finished`, `artifacts_verified`, `coverage_complete`, and
`analysis_complete`. Exit code zero alone satisfies none of the latter three.
Expected cells, exact update/token counts, final checkpoint readability, finite
metrics, target counts, unique identities, and fixture hashes must all be checked.

Use separate directories for smoke, final runs, and each failed attempt. Preserve
stdout/stderr, commands, timing, OOM/error messages, configuration, and partial
results. Never aggregate by loose filename globbing that includes smoke or obsolete
attempts. Select one accepted artifact per expected cell by manifest fingerprint.

Resume only with verified model, optimizer, RNG, scheduler, accumulation boundary,
and data-cursor state. Compare uninterrupted and interrupted/resumed smoke runs.
A resume that changes token exposure or optimizer state is a new training condition.
Infrastructure retries preserve the same seed/config and consume the separately
recorded reserve. A second infrastructure failure of the same run triggers an
investigation before further attempts. An authentic bad optimization outcome is
reported; repeated fresh starts to find a good outcome are forbidden.

Retain the final weights for all nine E1 runs and the initialization/metadata needed
to reconstruct their pairing. Retain a restart checkpoint and atomic latest pointer;
include sufficient optimizer/RNG/data state for recovery. Do not delete the only
reproducible final checkpoint to save storage. Pin existing source/adapted weights
and verify access before the deadline. Never overwrite another experiment's logs,
configs, tables, or checkpoint aliases.

## 8. Reporting acceptance

Each figure/table must trace to raw records, an aggregation script version, the
fixture and run manifests, and a coverage report. Recalculate from unrounded values;
round only for display. Validate plotted values and captions against the same data.
Report exact denominators, failed cells, calibration/adaptation budgets, dataset
IDs, hardware/backend differences, and limits of inference alongside the result.

If only part of a matched family completes, show the incomplete family explicitly
and make a correspondingly narrower claim. Do not call an exploratory subset the
completed planned experiment. Report every seed, corpus, and prespecified metric,
including cap harms and failed replications. Do not use "healthy", "stable", or
"unchanged dynamics" as labels derived only from a zero-input parameter scan.
