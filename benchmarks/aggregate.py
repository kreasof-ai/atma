"""Aggregate all structured benchmark logs into a tidy JSON/CSV matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path


MARKERS = {
    "retrieval": "===RETRIEVAL_RESULTS_JSON===",
    "babilong": "===BABILONG_RESULTS_JSON===",
    "base": "===BASE_RESULTS_JSON===",
    "longdoc": "===LONGDOC_RESULTS_JSON===",
    "serving": "===SERVING_RESULTS_JSON===",
}


def source_hash(path: Path) -> str:
    """Audit identity survives Git's Windows CRLF/Linux LF checkout conversion."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _extract(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    found = []
    for benchmark, marker in MARKERS.items():
        if marker not in text or "===END===" not in text.rsplit(marker, 1)[1]:
            continue
        block = text.rsplit(marker, 1)[1].split("===END===", 1)[0].strip()
        try:
            result = json.loads(block)
        except json.JSONDecodeError:
            continue
        found.append((benchmark, result))
    return found


def _model_name(result, path):
    cfg = result.get("model_config") or {}
    if cfg.get("is_foveal"):
        f_cfg = cfg.get("foveal_config") or {}
        core = cfg.get("attn_type") or f_cfg.get("attn_type") or "unknown"
        if core == "unknown":
            for c in ("polar", "nope", "rope"):
                if c in path.stem:
                    core = c
                    break
        mode = cfg.get("adaptation_mode") or f_cfg.get("adaptation_mode") or "local"
        return f"{core}_{mode}"
    name = cfg.get("arch_type") or cfg.get("attn_type")
    if name:
        return name
    for candidate in ("atma_raven_titans", "raven_native", "polar", "nope", "rope"):
        if candidate in path.stem:
            return candidate
    return "unknown"


def _row(model, benchmark, source, **values):
    return {"model": model, "benchmark": benchmark, "source_log": str(source), **values}


def _flatten(benchmark, result, source):
    model = _model_name(result, source)
    rows = []
    if benchmark == "retrieval":
        # The retired free-generation protocol produced uniformly zero scores for these
        # base checkpoints and could corrupt resumed matrices after the protocol migration.
        if result.get("protocol") != "teacher-forced-needle-v2":
            return rows
        suite = "synthetic" if result.get("haystack") == "synthetic-filler" else "real"
        primary_metric = (
            "token_accuracy"
            if result.get("protocol") == "teacher-forced-needle-v2"
            else "exact_match"
        )
        for task, lengths in result.get("results", {}).items():
            for length, depths in lengths.items():
                for depth, value in depths.items():
                    rows.append(_row(
                        model, benchmark, source, suite=suite, task=task, dataset=result.get("haystack"),
                        length=length, depth=depth, metric=primary_metric, value=value,
                        samples=result.get("num_samples"),
                    ))
        for field, metric in (
            ("exact_results", "exact_match"),
            ("nll_results", "nll_nats_per_token"),
        ):
            for task, lengths in result.get(field, {}).items():
                for length, depths in lengths.items():
                    for depth, value in depths.items():
                        rows.append(_row(
                            model, benchmark, source, suite=suite, task=task,
                            dataset=result.get("haystack"), length=length, depth=depth,
                            metric=metric, value=value, samples=result.get("num_samples"),
                        ))
        for cell in result.get("oom_cells", []):
            rows.append(_row(
                model, benchmark, source, suite=suite,
                task=cell.get("kind") or cell.get("task"),
                dataset=result.get("haystack"), length=cell.get("length"),
                depth=cell.get("depth"), metric="oom", value=True,
                samples=result.get("num_samples"),
            ))
    elif benchmark == "base":
        for task, metrics in result.get("results", {}).items():
            for metric in ("accuracy", "accuracy_norm", "target_nll", "target_perplexity"):
                if metrics.get(metric) is not None:
                    rows.append(_row(
                        model, benchmark, source, suite="zero_shot", task=task, dataset=None,
                        length=None, depth=None, metric=metric, value=metrics[metric],
                        samples=metrics.get("samples"),
                    ))
    elif benchmark == "longdoc":
        for dataset, dataset_result in result.get("results", {}).items():
            for length, metrics in dataset_result.get("lengths", {}).items():
                for metric in ("nll_nats_per_token", "perplexity", "bits_per_byte"):
                    if metrics.get(metric) is not None:
                        rows.append(_row(
                            model, benchmark, source, suite="fixed_target", task=None,
                            dataset=dataset_result.get("dataset_id", dataset),
                            length=length, depth=None, metric=metric,
                            value=metrics[metric], samples=metrics.get("documents"),
                        ))
                if metrics.get("oom"):
                    rows.append(_row(
                        model, benchmark, source, suite="fixed_target", task=None,
                        dataset=dataset_result.get("dataset_id", dataset),
                        length=length, depth=None, metric="oom", value=True,
                        samples=metrics.get("documents"),
                    ))
    elif benchmark == "serving":
        backend = result.get("backend")
        if backend == "direct-full-recompute":
            serving_suite = "direct_full_recompute"
        elif backend == "FovealLLM":
            serving_suite = "foveal_cached_generation"
        elif backend == "ordinary-engine-index-disabled":
            serving_suite = "ordinary_engine_index_disabled"
        elif (result.get("model_config") or {}).get("is_foveal"):
            serving_suite = "unverified_generation"
        else:
            serving_suite = "generation"
        for length, metrics in result.get("results", {}).items():
            for source_key, metric in (
                ("prefill_tokens_per_s", "prefill_tokens_per_s"),
                ("decode_tokens_per_s", "decode_tokens_per_s"),
                ("time_to_first_token_s", "time_to_first_token_s"),
                ("decode_latency_per_token_s", "decode_latency_per_token_s"),
                ("peak_allocated_bytes", "peak_allocated_bytes"),
                ("peak_reserved_bytes", "peak_reserved_bytes"),
                ("oom", "oom"),
            ):
                if source_key in metrics:
                    rows.append(_row(
                        model, benchmark, source, suite=serving_suite, task=None, dataset=None,
                        length=length, depth=None, metric=metric, value=metrics[source_key],
                        samples=metrics.get("samples"),
                    ))
        if result.get("max_successful_context_tokens") is not None:
            rows.append(_row(
                model, benchmark, source, suite=serving_suite, task=None, dataset=None,
                length=None, depth=None, metric="max_successful_context_tokens",
                value=result["max_successful_context_tokens"], samples=None,
            ))
    elif benchmark == "babilong":
        if result.get("protocol") != "heldout-short-finetune-v1":
            return rows
        for task, lengths in result.get("results", {}).items():
            for length, value in lengths.items():
                if value is None:
                    continue
                rows.append(_row(
                    model, benchmark, source, suite="adapted_reasoning", task=task,
                    dataset=result.get("dataset_id"), length=length, depth=None,
                    metric="accuracy", value=value,
                    samples=result.get("counts", {}).get(task, {}).get(length),
                ))
        for length, value in result.get("macro_average", {}).items():
            if value is not None:
                rows.append(_row(
                    model, benchmark, source, suite="adapted_reasoning", task=None,
                    dataset=result.get("dataset_id"), length=length, depth=None,
                    metric="macro_accuracy", value=value,
                    samples=result.get("macro_task_counts", {}).get(length),
                ))
        for cell in result.get("oom_cells", []):
            task = cell.get("task")
            length = cell.get("length")
            rows.append(_row(
                model, benchmark, source, suite="adapted_reasoning", task=task,
                dataset=result.get("dataset_id"), length=length, depth=None,
                metric="oom", value=True,
                samples=result.get("counts", {}).get(task, {}).get(length),
            ))
    for row in rows:
        row["protocol"] = result.get("protocol")
        row["backend"] = result.get("backend") or result.get("generation_backend")
    return rows


def aggregate(log_dir: Path):
    rows = []
    sources = []
    latest = {}
    manifest_path = log_dir / "aggregation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    selections = {}
    if manifest is not None:
        for entry in manifest["sources"]:
            path = (log_dir / entry["path"]).resolve()
            if not path.is_relative_to(log_dir.resolve()):
                raise ValueError("aggregation source must be inside log_dir")
            if path in selections:
                raise ValueError(f"duplicate selected source: {path}")
            if source_hash(path) != entry["sha256"]:
                raise ValueError(f"selected source changed since audit: {path}")
            selections[path] = entry
        paths = sorted(selections)
    else:
        paths = sorted(log_dir.rglob("*.log"))
    for path in paths:
        if path.name.endswith(".console.log"):
            continue
        if path.name.startswith("smoke_"):
            continue
        extracted = _extract(path)
        selection = selections.get(path.resolve())
        if selection:
            extracted = [(b, r) for b, r in extracted if b == selection["benchmark"]]
            if not extracted:
                raise ValueError(f"selected source has no complete result: {path}")
            for _, result in extracted:
                if selection.get("backend_override"):
                    result["backend"] = selection["backend_override"]
        if not extracted:
            continue
        logical_stem = re.sub(r"\.attempt-\d+$", "", path.stem)
        logical_path = path.with_name(logical_stem + path.suffix)
        previous = latest.get(logical_path)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            latest[logical_path] = (path, extracted)

    for path, extracted in sorted(
        latest.values(), key=lambda item: str(item[0])
    ):
        for benchmark, result in extracted:
            flattened = _flatten(benchmark, result, path)
            if flattened:
                sources.append({"path": str(path), "benchmark": benchmark})
                rows.extend(flattened)
    rows.sort(key=lambda row: (
        row["model"], row["benchmark"], str(row.get("suite")), str(row.get("task")),
        str(row.get("dataset")), str(row.get("length")), str(row.get("depth")), row["metric"],
    ))
    if manifest is not None:
        keys = [tuple(str(row.get(k)) for k in (
            "model", "benchmark", "suite", "task", "dataset", "length", "depth", "metric"
        )) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("selected experiments contain duplicate result cells")
    return {
        "schema_version": 2,
        "generated_at_unix": int(time.time()),
        "log_dir": str(log_dir),
        "selection_manifest": str(manifest_path) if manifest is not None else None,
        "sources": sources,
        "rows": rows,
    }


def write_outputs(result, json_path: Path, csv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = [
        "model", "benchmark", "suite", "task", "dataset", "length", "depth",
        "metric", "value", "samples", "source_log", "protocol", "backend",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate ATMA structured benchmark logs.")
    ap.add_argument("--log_dir", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args(argv)
    log_dir = args.log_dir.resolve()
    result = aggregate(log_dir)
    json_path = args.out_json or log_dir / "benchmark_matrix.json"
    csv_path = args.out_csv or log_dir / "benchmark_matrix.csv"
    write_outputs(result, json_path, csv_path)
    print(f"[aggregate] {len(result['rows'])} rows -> {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
