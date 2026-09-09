"""Validated paper summaries of the completed, unclamped TDA benchmark bundle."""

import json
from statistics import mean

from re_evaluation_data import ROOT, LENGTHS, BABI_LENGTHS, DEPTHS, DATASET_IDS, TASK_METRICS


def load_tda():
    path = ROOT / "supplementary/robustness/work/evaluation/tda_pipeline/benchmark_matrix.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]

    def select(benchmark, metric, **filters):
        return [r for r in rows if r["benchmark"] == benchmark and r["metric"] == metric
                and all(r[k] == v for k, v in filters.items())]

    def one(benchmark, metric, **filters):
        found = select(benchmark, metric, **filters)
        if len(found) != 1:
            raise ValueError((benchmark, metric, filters, len(found)))
        return float(found[0]["value"])

    retrieval = {}
    retrieval_by_depth = {length: {} for length in LENGTHS}
    for length in LENGTHS:
        for depth in DEPTHS:
            cells = select("retrieval", "token_accuracy", length=length)
            cells = [r for r in cells if str(r["depth"]) == depth]
            assert len(cells) == 4 and {(r["suite"], r["task"]) for r in cells} == {
                (s, t) for s in ("synthetic", "real") for t in ("niah", "passkey")}
            retrieval_by_depth[length][depth] = mean(r["value"] for r in cells)
    for metric in ("token_accuracy", "exact_match"):
        retrieval[metric] = {}
        for suite in ("synthetic", "real"):
            retrieval[metric][suite] = {}
            for length in LENGTHS:
                cells = select("retrieval", metric, suite=suite, length=length)
                expected = {(t, d) for t in ("passkey", "niah") for d in DEPTHS}
                assert len(cells) == 6 and {(r["task"], str(r["depth"])) for r in cells} == expected
                assert all(r["samples"] == 50 for r in cells)
                retrieval[metric][suite][length] = mean(r["value"] for r in cells)
        retrieval[metric]["overall"] = {
            length: mean(retrieval[metric][s][length] for s in ("synthetic", "real"))
            for length in LENGTHS}
    bpb = {d: {length: one("longdoc", "bits_per_byte", dataset=dataset, length=length)
               for length in LENGTHS} for d, dataset in DATASET_IDS.items()}
    for d, dataset in DATASET_IDS.items():
        assert all(r["samples"] == (8 if d == "finepdfs" else 5)
                   for r in select("longdoc", "bits_per_byte", dataset=dataset))
    bpb["mean"] = {length: mean(bpb[d][length] for d in DATASET_IDS) for length in LENGTHS}
    babi = {length: one("babilong", "macro_accuracy", length=length) for length in BABI_LENGTHS}
    downstream = {task: 100 * one("base", metric, task=task) for task, metric in TASK_METRICS.items()}
    serving_metrics = ("time_to_first_token_s", "prefill_tokens_per_s", "decode_latency_per_token_s",
                       "decode_tokens_per_s", "peak_allocated_bytes", "peak_reserved_bytes")
    serving = {length: {m: one("serving", m, suite="direct_full_recompute", length=length)
                        for m in serving_metrics} for length in LENGTHS[:-1]}
    assert all(one("serving", "oom", suite="direct_full_recompute", length=l) == 0
               for l in LENGTHS[:-1])
    return dict(retrieval=retrieval, retrieval_by_depth=retrieval_by_depth,
                bpb=bpb, babi=babi, downstream=downstream, serving=serving)
