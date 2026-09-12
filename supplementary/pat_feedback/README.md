# PAT feedback: experiment and evidence plan

**Protocol v1, 2026-09-12. Status: specified; no experiments launched or completed by this plan.**

This directory defines the follow-up to the ICLR PAT report discussed on September
12. The objective is to settle Polar attribution, test the limits of the retention
diagnostic, and improve evaluation coverage. A successful experiment is a complete,
valid comparison, including a null or unfavorable result.

The plan builds on the existing [robustness supplement](../robustness/README.md),
[gamma diagnostic](../../gamma_diagnostics/README.md), and
[Foveal CPT](../../foveal_cpt/README.md). Those artifacts remain historical evidence;
their results must not be silently replaced by runs from this directory.

## Read and use

- [EXPERIMENTS.md](EXPERIMENTS.md): hypotheses, exact comparisons, measurements,
  completion criteria, and interpretation for E0-E6.
- [RIGOR.md](RIGOR.md): required controls, numerical standards, statistical units,
  provenance, failure handling, and claim boundaries.
- [EXECUTION.md](EXECUTION.md): implementation order, readiness evidence, scheduling,
  artifact contracts, and the final handoff checklist.
- [plan.json](plan.json): machine-readable scope, run matrix, exact token counts,
  endpoints, and unresolved readiness items. This is a planning manifest, **not a
  runnable training configuration**.

From the repository root:

```bash
python supplementary/pat_feedback/validate_plan.py
python supplementary/pat_feedback/validate_plan.py --require-ready
```

The first command checks the plan's internal consistency and local source links.
The second also rejects unresolved readiness items. It is expected to fail for the
initial plan. Neither command runs models, allocates GPUs, validates scientific
results, or certifies a numerical implementation.

## Required scope

| ID | Priority | Work | New training | Main deliverable |
|---|---|---|---|---|
| E0 | Prerequisite | Polar numerical and checkpoint-path audit | None | Oracle/kernel decision and measured impact |
| E1 | Core | NoPE / temperature-softmax / Polar, all with memory | 3 arms x 3 initializations | Component-attribution contrasts |
| E2 | Core | Original / Foveal CPT x uncapped / capped, for NoPE and Polar | None | Eight-condition retention comparison |
| E3 | Core | Runtime retention, state, and readout diagnostics on E2 | None | Evidence separating candidate mechanisms |
| E4 | Core | Independent-document retrieval, including fresh replications | None | Document-level paired retrieval evidence |
| E5 | Conditional | Cap sensitivity and alternative-head controls | None | Diagnostic specificity and trade-offs |
| E6 | Conditional | Warmed, repeated serving measurements | None | Comparable implementation timing distributions |

E0 blocks the production comparisons whose outputs depend on its unresolved
semantics. Dataset preparation, protocol implementation, and source inspection can
proceed meanwhile. E3 consumes the E2 checkpoint/condition definitions. E4 defines
the common retrieval fixtures for E1 and E2 as well as its own replication panel.
E5-E6 follow the core work and may remain prepared for rebuttal.

## Budget and timing

E1 consists of nine fresh runs, each **1,900 optimizer updates x 524,288 target
tokens = 996,147,200 training tokens**. The successful-run total is
**8,965,324,800 tokens**, approximately 9B. All three arms of an initialization are
an indivisible comparison group. There is no best-seed selection or replacement of
a scientifically unfavorable seed.

The historical Full Polar pilot took 17,677 seconds of training, approximately
4.91 hours. Nine such runs would take about **44.2 GPU-hours of training**. This is
an extrapolation, not a reservation or an end-to-end estimate. Evaluation,
implementation, compilation, instrumentation, failures, and checkpoint storage are
additional. The GPU-hour allocation and retry reserve remain unresolved until a
measured capacity record exists; no unlimited retry allowance is implied.

The full core retrieval plan contains **42,120 distinct model-prompt evaluations**
after reusing the two original uncapped conditions shared by E2 and E4. This counts
21 model/condition identities: nine E1 models at five lengths, and twelve E2/E4
conditions at six lengths. It does not include BPB, numerical probes, or optional
experiments. Profile long-context evaluation before committing to a calendar.

| Target dates, 2026 | Work and exit condition |
|---|---|
| Sep 12-13 | Implement missing control and manifests; resolve E0; freeze protocols and measure capacity |
| Sep 13-16 | Run complete E1 groups; evaluate E2/E4 on frozen fixtures; save predictions and hashes |
| Sep 16-18 | Complete E3; validate coverage; analyze all results; settle the abstract's supported claims |
| Sep 19-25 | Finish full-paper validation and writing; conditional E5/E6 if capacity permits |

The abstract deadline is **September 18, 23:59 AoE** (September 19, 18:59 WIB);
the full paper deadline is **September 25, 23:59 AoE** (September 26, 18:59 WIB).
Titles and abstracts may be refined before the full submission deadline while
remaining consistent with the original submission. Sources checked September 12:
[ICLR call for papers](https://iclr.cc/Conferences/2027/CallForPapers) and
[author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
Do not delay the abstract submission waiting for an experiment.

## Decisions already made

- Keep the gamma contribution diagnostic. A long zero-input horizon is not a
  sufficient predictor of failure; Foveal adaptation is relevant counterevidence.
- No bounded-gamma retraining, larger model, new external architecture, new
  BABILong adaptation, or additional full 120-cell sweep belongs to this scope.
- A 9.816B temperature-control comparison is a separately budgeted extension.
  It cannot be substituted for an unfinished 1B comparison or presented as already
  scheduled. A cooled-down 1B checkpoint is not automatically a matched 9.816B run.
- Numerical conventions, evaluation fixtures, and primary contrasts are fixed
  before production outcomes are inspected. Later amendments are explicit and
  preserve the original result ledger.
- Existing Foveal/PAT observations informed this design. This is a prospective
  follow-up protocol, not a claim that those observations were preregistered.

## Current implementation gaps

The temperature-softmax arm and common-tensor initialization check still need
implementation. Exact checkpoint/tokenizer/training-shard pins, the independent
document fixture, a complete launch budget, and E0/E3 parity evidence still need
resolution. Existing scripts must not be treated as supporting these features
merely because a config flag can be added.

The old `data_seed` field does not shuffle the training stream; E1 explicitly uses
three initializations on the same frozen data order. The Foveal loader uses packed
token streams without document-boundary resets. Both facts must survive into the
methods description. See [RIGOR.md](RIGOR.md) for the associated controls.
