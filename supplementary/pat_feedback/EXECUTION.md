# Execution and evidence handoff

This is the implementation order for the specified study. The current directory
contains a protocol and a structural validator, not training/evaluation launchers.
There are no completed experiment results or ready GPU jobs in this initial state.

## 1. Establish one operational tree

When implementation begins, keep the resolved executable configs only in
`supplementary/pat_feedback/configs/`. Do not create another competing config tree
under `work/`, alter the existing robustness configs, or pass `plan.json` to a
training worker. Every config points to a plan experiment ID and protocol hash.

Use these namespaces when jobs exist:

```text
supplementary/pat_feedback/
  configs/                         # One authoritative executable config tree
  manifests/                       # Input pins, prompts, documents, expected cells
  work/smoke/<fingerprint>/         # Infrastructure-only fixtures and output
  work/runs/<run_id>/<attempt_id>/  # Logs, provenance, status, validation
  work/evaluation/<fingerprint>/   # Raw predictions and per-document scores
  work/diagnostics/<fingerprint>/  # Numerical and telemetry evidence
  work/analysis/<protocol_hash>/   # Coverage reports and generated tables/figures
  records/                        # Frozen launch evidence and amendments
checkpoints/pat_feedback/          # Final and restart weights, stored separately
```

These are future output locations, not an assertion that those artifacts exist.
Store portable repository-relative artifact paths in manifests. Large weights may
live on pinned remote storage, but their revision/file hashes and retrieval recipe
must be retained. Use atomic file replacement for status and checkpoint pointers.

## 2. Resolve readiness with evidence

The `readiness` entries in `plan.json` begin unresolved. Mark an item satisfied only
after its specified evidence exists, has been checked, and is referenced by a
repository-relative file path. No placeholder strings or statements of intent count
as evidence. The structural validator checks that referenced evidence files exist;
it cannot judge their scientific validity or replace GPU tests.

| Readiness item | Required evidence before dependent production work |
|---|---|
| `operator_semantics` | E0 equations/floors, CPU/GPU test results, checkpoint-path impact, execution revision |
| `temperature_control` | Implemented train-time arm; t=1 operator/shared-gradient parity; learned-alpha gradient and optimizer membership |
| `matched_initialization_and_recipe` | Nine resolved configs, post-initialization shared-tensor hashes, full optimizer/schedule/batch audit |
| `immutable_inputs` | Full checkpoint/dataset/tokenizer/dependency revisions and file hashes; actual loader/backend verification |
| `evaluation_fixtures` | Frozen BPB/document/prompt manifests, independent real-document identities, scoring/aggregation tests |
| `runner_integrity` | Expected-cell validator, unique artifacts, strict failures, atomic/resumable execution, interrupted-resume test |
| `capacity_and_storage` | Actual GPU allocation, short and 256K timings, training/evaluation/retry budgets, storage and retrieval checks |
| `instrumentation_parity` | E3 normal/instrumented output comparison, coverage limits, measured overhead |

Use readiness per dependency: E1 does not need to wait for E3 instrumentation;
E2/E4 do not need a completed new temperature arm. All depend on valid relevant
numerical paths, immutable inputs, fixtures, runner integrity, and measured capacity.
`--require-ready` is deliberately a whole-core-plan check, not a scheduler.

## 3. Implementation sequence

1. Pin the inspected baseline and implement the E0 tests. Preserve historical
   semantics while deciding the new comparison's finite-precision convention.
2. Build the temperature arm and the shared-initialization mapping. Export the
   complete recipe. Implement explicit update and consumed-target-token counting;
   do not infer the budget from the nominal number of 100M-token shards.
3. Resolve existing checkpoint sources and produce BPB/retrieval manifests.
   Implement per-trial output and paired aggregation before final evaluation.
4. Generate nine E1 configs and E2/E4 expected-cell manifests from the frozen plan.
   Check shared cell deduplication. Implement a read-only coverage validator that
   rejects missing/duplicate cases, null/NaN metrics, mismatched hashes, smoke
   contamination, and incomplete conditions.
5. Run separate infrastructure smoke tests for all three attention arms and all
   relevant original/Foveal capped paths. Check short and maximum context; verify
   remote routing and actual cap binding. Warm compilation before estimating time.
6. Measure a representative evaluation cell at each required length and a complete
   training update for every arm. Capture instrumentation/storage costs separately.
   Freeze a feasible capacity record before production; use the same longest-context
   protocol for every compared model even if batching must vary for evaluation.
7. Dispatch complete initialization triplets and complete checkpoint-condition
   pairs. Every worker records a unique claim with host/PID/job ID, config fingerprint,
   status, and timestamps. Check worker liveness before clearing a stale claim.
8. Validate artifacts before aggregating; inspect every initialization and corpus.
   Finish E3 on its frozen subset. Add E5/E6 only within the conditional budget.

Test teacher-forced retrieval with a tiny known-answer fixture that catches target
offset, masking, and denominator errors. Test BPB aggregation with differing byte
counts to catch accidental averaging of ratios. Test paired resampling by preserving
whole document clusters. These are functional checks, not copies of implementation
formulas that could repeat the same bug.

## 4. Capacity and stop rules

E1 successful-run budget: nine x 996,147,200 = **8,965,324,800 tokens**. Record
additional smoke/replayed/failed-attempt tokens and GPU-hours separately; they do
not disappear from the compute ledger. The initial plan has no allocated retry
reserve or GPU-hour ceiling. Resolve and record finite values before launch.

The capacity record must include available GPU-hours by date/device, per-arm warmed
step timings, per-length evaluation timings, expected job counts, model reload and
compile overhead, trace overhead, storage, failed-attempt reserve, and manuscript
handoff time. Use a conservative projection from measured throughput, not the
historical 44.2-hour training estimate as the total project cost.

If capacity is insufficient, postpone E5/E6 first. Then record a reduced-scope
amendment before outcomes are inspected: reduce whole matched groups or a common
evaluation grid, and update claims and expected counts together. Never drop the
slow/failing arm alone, remove difficult documents after seeing scores, or call two
completed initializations the planned three-seed study. If the predeclared core
cannot complete, preserve the incomplete plan and report the available evidence
explicitly. The deadline does not waive matching or numerical correctness.

Pause an affected family for missing pins, unresolved E0 semantics, invalid cap
binding, fixture mismatches, repeated infrastructure failure, or projected budget
overrun. An accuracy loss or absence of a Polar advantage is not a stop/retry rule.

## 5. Required records

### Per training attempt

Record `run_id`, `attempt_id`, experiment/protocol/config hashes, initialization
seed, effective data-order identity, shared-init hashes, exact model parameters,
optimizer groups/hyperparameters, hardware/runtime, command, start/end times,
completed updates, consumed/scored/replayed token counts, data cursor, RNG state,
checkpoint hashes, exit status, and artifact-validation status. Preserve loss and
gradient logs; use the final prespecified checkpoint for primary evaluation.

### Per evaluation case

Record model/condition fingerprint, checkpoint and fixture hashes, corpus and
document/case ID, task, suite, depth, requested/actual context lengths, insertion or
target offset, prompt/target token hashes, target IDs, predicted IDs, position-level
correctness and exact correctness; or NLL sum/target bytes/target tokens for BPB.
Include actual intervention target, configured/final cap, runtime/backend, route
configuration, success/error/OOM status, elapsed time, and output provenance.

### Per analysis release

Record expected and observed cells, exclusions with reasons, accepted-attempt map,
raw input hashes, aggregation/plotting revision, bootstrap seed and cluster unit,
full-precision statistics, per-seed/corpus tables, rounding policy, and proposed
claim text. Generate all plots and tables from those accepted raw records.

## 6. Handoff checklist

- [ ] E0 decision is explicit; historical and new operator versions are separated.
- [ ] All nine E1 final checkpoints have matched initialization/recipe evidence.
- [ ] E2 has all eight conditions on the same 18 fixed-target documents and prompt set.
- [ ] E3 reports measured runtime behavior, parity limits, and unmeasured quantities.
- [ ] E4 contains all six checkpoints and all 30 independent real documents.
- [ ] Coverage checks exclude smoke/duplicates and expose failures without zero-filling.
- [ ] Tables use raw values; document-level uncertainty and training-seed limits are explicit.
- [ ] Foveal packed-stream training, prior selection, adaptation budget, and cap harms are disclosed.
- [ ] Every claim is traceable to a figure/table and every figure/table to raw artifacts.
- [ ] Abstract wording uses completed evidence; 1B results are not promoted to 9.816B claims.
- [ ] Conditional experiments are identified as completed, deferred, or not run.
- [ ] The author reviews interpretations before manuscript integration; routine
      protocol implementation and reversible local fixes need no extra approval ceremony.

Executing the study requires the missing runners and evidence records above.
This initial delivery specifies and checks the plan; it does not dispatch jobs.
