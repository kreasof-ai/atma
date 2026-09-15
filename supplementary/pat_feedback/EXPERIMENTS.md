# Required experiments

All outcomes below are prospective. Existing results are context, not completed
cells in this plan unless exact input, checkpoint, code, and condition fingerprints
permit reuse. Counts and fixed choices are recorded in [plan.json](plan.json).

## Shared fixtures

### Retrieval

Freeze 30 unique, sufficiently long FinePDFs documents from a pinned dataset
revision. Select them by a recorded seeded permutation of an eligible pool, without
model scoring; preserve pool construction, scan limits, exclusions, and hashes.
Exclude documents already used for implementation tuning and the archived BPB panel
where identities can be recovered. Record unknown pretraining overlap rather than
claiming guaranteed training-data independence. If fewer than 30 qualify, record
the shortfall and amend the design before any production scoring.

Each document defines one case ID and a five-token target, used across passkey and
NIAH, three depths (0.1, 0.5, 0.9), every length, and every model. Store actual target
token IDs and exact prompt hashes; enforce five tokens after tokenization. Preserve
the existing insertion and depth definitions and record final prompt length and
insertion offset. Prevent accidental target duplication in the haystack. The real
context must come from that document, not a shared concatenated stream. Shorter
contexts use nested prefixes of the same case.

Create a paired synthetic case for each of the 30 case IDs using the same target
and placement convention. Synthetic cases share a filler generator; they are not
30 independently sampled natural documents.

Use lengths 2K/8K/16K/32K/64K/256K, with K=1024. E1 stops at 64K in the core plan;
256K for the new 1B models requires a resource amendment before results are seen.
There are 2 tasks x 3 depths x 30 cases = 180 prompts per suite and length, hence
1,800 prompts per E1 model and 2,160 per E2/E4 condition. Score five teacher-forced
argmax positions and all-five-correct exact match; save correctness vectors.
Do not call either metric autoregressive generation accuracy.

### Fixed-target likelihood

Recover and freeze the original paper's **18-document panel**: eight FinePDFs,
five PG-19, five Proof-Pile. All lengths predict the same 256-token target span
immediately following token 262,144, using a context of the specified length ending
at that target. Store document, context, target-token, and target-byte hashes.
If this exact panel cannot be recovered, construct a replacement manifest and rerun
every compared checkpoint on it; never combine archived and replacement aggregates.

Dataset BPB = summed target NLL in bits / summed UTF-8 target bytes. The cross-dataset
mean weights the three dataset BPBs equally. Score 4,608 target tokens per model and
length, and retain per-document NLL sums, token counts, and byte counts. E1 uses its
five lengths; E2 uses all six. Smoke documents are separate from both final fixtures.

## E0 - Numerical semantics and checkpoint-path audit

**Question:** Are the stated Polar operator, numerical oracle, training kernel,
and checkpoint evaluation paths equivalent in the regimes relevant to the claims?

Required checks:

1. Direction-floor semantics: with U = Zs, compare U/max(norm(U), epsilon*Z)
   against the current streaming U/max(norm(U), epsilon). Include zero vectors,
   cancellation, tiny nonzero norms, null vectors equal to zero/nonzero, and a sweep
   of Z. Verify forward values and backward behavior away from nondifferentiable
   threshold boundaries.
2. Participation ratio: audit materialized real-mass renormalization, Q2 floors,
   and null confidence. Include dominant null logits, underflow, uniform real
   weights, one real key, and all-masked/padding queries. Define the all-null
   convention explicitly. Do not assume every path uses the same Q2 floor.
3. Cover dense, block-streaming, selected-page sparse, and decode paths that will
   actually be used, plus causal masks, lengths, GQA mapping, FP32 and BF16, and
   relevant gradients. Verify that the temperature control with t=1 recovers the
   ordinary NoPE operator and shared gradients.
4. Run checkpoint probes that record normalized/unnormalized direction norms,
   real/null mass, floor activation rates, logit differences, and loss differences.
   Include original and Foveal Polar; identify which historical implementation each
   checkpoint used. Long-sequence smoke probes are infrastructure tests, not final
   test-document outcome selection.

**Completion:** numerical decision, explicit formulas/floors, test artifacts,
per-path tolerances, checkpoint impact, and a frozen execution revision. Historical
checkpoints retain their original semantics for forensic reproduction. If the
operator changes, label the new operator/version and rerun all affected comparisons;
do not transfer a corrected inference result into an archived table silently.

An edge-case mismatch can be real without materially changing benchmark results.
Measure that distinction; do not assert zero impact from random-input parity alone.

## E1 - Temperature-matched component attribution

**Primary question:** Does Full Polar improve natural-text exact retrieval over
ordinary softmax equipped with the same learned length-temperature family?

| Arm | Attention output | Recurrent memory |
|---|---|---|
| `nope_memory` | Ordinary NoPE softmax | Enabled, unchanged |
| `temperature_softmax_memory` | Ordinary NoPE softmax using t(n)=1+softplus(alpha)*log(n) | Enabled, unchanged |
| `polar_memory` | Full Polar, with the E0-resolved convention | Enabled, unchanged |

The temperature control adds only one learned alpha per query head, initialized
to -1.0, matching Full Polar. It has no Polar null competitor, direction
normalization, or magnitude channel. It retains NoPE's Q/K preprocessing, Canon
convolutions, GQA, output gate/projection, and memory. Use the true visible causal
key count n=i+1. Do not add a temperature only at inference to a trained NoPE model.

Train a fresh triplet at each initialization seed **20270912, 20270913, 20270914**.
The three arms in a triplet must have identical common tensors after every
initialization and projection-zeroing operation, verified by a named tensor map
and checksums. Equal RNG seeds alone are insufficient when constructors consume
different random draws. Initialize new alpha identically in the two relevant arms.
Keep all three initializations on the same frozen training token order. No data-order
replication is claimed.

Use the matched 378M ATMA shape (16 layers, width 1024, eight query/memory heads,
two KV heads, head dimension 128), 2,048-token training context, full causal attention,
memory enabled, no distractor alignment and no representation regularization.
Global batch: 524,288 tokens; microbatch: four sequences; accumulation: 64
microbatches per update on one GPU. Train exactly 1,900 updates. The checkpoint
used for the main comparison is the final update, not the best retrieval checkpoint.

Start from the archived component recipe. Preserve its Muon/AdamW parameter split,
learning rates, normalization, gradient clipping, projection initialization,
validation token stream, and 0.7 cooldown fraction. Export the complete resolved
optimizer configuration and parameter membership before launch. The new alpha must
enter the same scalar-parameter optimizer group as Polar's alpha. Assert that every
trainable parameter appears exactly once; log exact parameter counts and differences.
Do not silently switch all parameters to AdamW to simplify implementation.

**Primary endpoint:** natural-text exact retrieval, equally averaged over
8K/16K/32K/64K, both tasks, three depths, and 30 documents. The primary contrast is
Polar minus temperature-softmax. Report one contrast per initialization, the mean,
and the range. Secondary: temperature-softmax minus NoPE, each length separately,
synthetic exact retrieval, both token-retrieval curves, 64K BPB, and 2K validation
NLL. Do not hide loss/retrieval trade-offs behind a composite quality score.

**Completion:** all nine final checkpoints and complete required evaluation cells,
with provenance and raw predictions. An authentic optimization divergence is a
result to report, not a seed to replace; it prevents claiming a complete finite
nine-checkpoint comparison. Infrastructure failures remain explicitly incomplete.

**Interpretation:** an advantage supports the full package over this particular
temperature control at this budget. It does not prove each Polar channel necessary,
establish a memory-off interaction, or extend the conclusion to 9.816B training.
Near-zero, mixed, or unfavorable contrasts leave incremental benefit unresolved or
unsupported on the tested endpoint; preserve all of them.

## E2 - Adaptation changes the effect of retention capping

**Question:** Does the outcome of a fixed retention intervention depend on whether
the checkpoint has undergone Foveal adaptation?

For NoPE and Polar, compare original Stage II and existing Foveal `lm_output_kl`
checkpoints, each uncapped and capped. This is **2 cores x 2 adaptation states x
2 cap conditions = 8 conditions**, with no new training. Pin every checkpoint to
an immutable revision and a weights hash. Re-evaluate all eight conditions on the
shared fixtures rather than comparing the historical 18-document and 12-document
BPB aggregates.

Select the original outlier by the largest total zero-input retention logit,
including the fixed config bias, with a deterministic lowest-block/head tie break.
Track that same zero-based layer/head in the adapted checkpoint, and separately
report its rank after adaptation. This defines a comparison of the same inherited
head. If the adapted maximum moves, report it; a new maximum-head intervention is
secondary and cannot replace the fixed-head primary comparison. Expected original
targets are NoPE B2/H5 and Polar B2/H6, subject to verification from the pinned files.

Apply min(z, logit(2^(-1/256))) to the **final activation-conditioned logit**,
including the fixed +3.9 where configured. Record actual resolved module paths,
target cardinality, ceiling, dtype, and hook binding. Check wrapped Foveal paths.
The unselected heads and checkpoint tensors must remain unchanged, and removing
the intervention must restore the original forward outputs within E0 tolerance.

**Primary endpoint:** 256K mean BPB. Define cap benefit as
B = BPB(uncapped) - BPB(capped), separately for original and adapted models.
Report B(original) - B(adapted) per core, along with all four raw values and the
2K-to-256K growth within each condition. Secondary: each corpus, intermediate
lengths, exact/token retrieval split by haystack, and native-context loss changes.

**Completion:** complete eight-condition BPB and retrieval matrices, all-head
zero-input scans, matching fixture hashes, and paired capped/uncapped execution
evidence. Alternate condition order by case to avoid a systematic order effect.

**Interpretation:** modest degradation with a persistent parameter outlier shows
that this operating point is insufficient to predict catastrophic failure. Residual
cap benefits and retrieval harms remain part of the result. Foveal changes attention,
training length, token exposure, and backbone weights; this experiment does not
isolate sparsity or establish unchanged runtime memory dynamics. A sufficient
contraction bound need not be a necessary condition for empirical performance.

BABILong is outside the immediate eight-condition rerun. Its existing adapted
results may be cited with their own protocol limits. Any new BABILong comparison
requires separately pinned final QA-adapted checkpoints, task rows, prompts, and
fresh retention scans; base checkpoints cannot stand in for adapted ones.

## E3 - Runtime retention and use of the outlier head

Use all eight E2 conditions on a prespecified six-document subset (two per BPB
corpus), at 2K/64K/256K. No document is selected for an especially large cap gain.

Record final pre-cap logits, post-cap logits, stable log-gamma, and relevant
quantiles/exceedance rates for every head; separate parameter-only H0 from runtime
summaries. Calculate half-life with stable log-sigmoid/expm1 arithmetic rather than
rounding sigmoid(z) to 1. Also record actual backend dtype/saturation: the real-number
sigmoid bound does not guarantee that a finite-precision gate is strictly below 1.

For the tracked head and prespecified same-layer controls, record write gates,
state Frobenius norms, readout norms before/after RMS normalization, output gates,
and the projected per-head contribution to the residual. Attribute projection
columns by head and keep the shared output bias separate; verify that contributions
sum back to the normal memory branch. Sample state summaries at fixed boundaries
without resetting state, cropping history, or changing route support.

Eager/reference instrumentation must pass parity against the production path on
supported overlap lengths. If full-length state tracing is infeasible, retain
production-compatible gate/readout traces and report the state trajectory as
unmeasured. Do not silently replace the production kernel with a different operator.

**Completion:** instrumentation parity evidence, all sampled-condition traces,
coverage and overhead measurements, and a figure separating:

- changed activation-conditioned retention;
- changed writing/state accumulation;
- changed readout or reliance on the head.

Trace correlations suggest mechanisms; they do not uniquely identify a causal
path. A readout norm is not a task-level causal attribution. Persistent H0 alone
does not establish persistent runtime retention or an unchanged memory module.

## E4 - Independent-document retrieval and replication

Evaluate six untouched checkpoints: original NoPE/Polar and both existing fresh
NoPE/Polar initialization pairs. Use the shared 30-document fixture and six lengths.
Reuse original uncapped results from E2 only when complete fingerprints match;
the four fresh checkpoints add four model-condition evaluations. Foveal results
remain in their separate adapted group. New 1B models remain the E1 group.

**Primary endpoint:** paired Polar-minus-NoPE natural-text exact accuracy averaged
over 8K/16K/32K/64K, reported separately for the primary pair and each fresh pair.
Also show the 256K exact result explicitly even when it is zero, every length curve,
token accuracy, and synthetic results. Report the original-versus-fresh microbatch
difference rather than interpreting the three pairs as an isolated seed sweep.

**Completion:** six-checkpoint coverage, 30 independently sampled real documents,
per-trial target/prediction arrays, and paired document-cluster uncertainty. Both
tasks, all depths, and nested lengths from one document remain in the same cluster.

This tests breadth across sampled documents. It does not establish generalization
to every corpus, instruction following, or autoregressive retrieval. Any comparison
with the old single-stream protocol is descriptive and uses distinct labels.

## E5 - Conditional sensitivity and specificity

Use the same six-document diagnostic subset at 2K/64K/256K and four checkpoints
(original/adapted NoPE and Polar). Prespecify the grid before scoring:

- inherited outlier head: no cap, H=128, H=256, H=512, H=1024;
- H=256 on the highest-H0 other head in that layer;
- H=256 on one other same-layer head selected by a recorded fixed RNG seed,
  excluding the outlier and the preceding control.

Deduplicate reused baseline/H=256 cells by fingerprint. Report cap-binding rates
and task losses: a cap that never binds is not a strength-matched control. If no
alternative cap binds, that limitation is the result; do not retune ceilings after
seeing task outcomes. Readout ablation would be a different intervention requiring
an explicit extension. No global cap is silently substituted for the one-head test.

**Deliverable:** complete prespecified sensitivity curves, individual corpus/task
effects, and all 2K loss shifts. The historical 0.05-nat native-loss guardrail may
be displayed as a reference line; it is not a rule for deleting unfavorable cells.
Do not select a new best ceiling on these documents and report it as held-out proof.

## E6 - Conditional serving repeatability

Use original NoPE/Polar/RoPE, Raven Native, Atma-Raven-Titans, and TDA hybrid at
2K/32K/128K, one sequence and 32 generated tokens. Keep weights, backend, dtype,
GPU, tokenized prompt, and decoding settings pinned. Confirm the backend executes
the claimed operator; sparse Foveal speed is excluded until remote routing and
index-output equivalence are separately verified for its serving implementation.

Measure cold-start/compile time separately. Per model/shape, complete untimed
shape-specific warmup, then at least ten timed repetitions with GPU synchronization
or CUDA events as appropriate. Interleave model/shape order across rounds where
practical; record clocks, competing workload, and unexpected recompilation. Preserve
every repetition and a preset invalid-measurement rule.

Report median, IQR, and min/max for prefill or TTFT, subsequent-token latency and
throughput, with generated-token counts. TDA's full-prefix recomputation remains
explicitly labeled. Report allocated/reserved/pool/occupied-KV memory separately
when shown; none of these measurements isolates architecture-only speed.
