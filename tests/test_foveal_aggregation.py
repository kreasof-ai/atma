import json
from pathlib import Path

import pytest

from benchmarks.aggregate import aggregate, _flatten, source_hash


def write_log(path, value=90):
    result = {"protocol": "teacher-forced-needle-v2", "haystack": "synthetic-filler",
              "num_samples": 10, "model_config": {"is_foveal": True, "attn_type": "nope", "adaptation_mode": "kl"},
              "results": {"passkey": {"2k": {"0.1": value}}}}
    path.write_text("===RETRIEVAL_RESULTS_JSON===\n" + json.dumps(result) + "\n===END===\n")


def select(folder, *paths):
    manifest = {"sources": [{"path": p.name, "benchmark": "retrieval",
                             "sha256": source_hash(p)} for p in paths]}
    (folder / "aggregation_manifest.json").write_text(json.dumps(manifest))


def test_manifest_excludes_smoke_and_reruns_and_checks_integrity(tmp_path):
    full = tmp_path / "retrieval_full.log"
    smoke = tmp_path / "smoke_partial.log"
    rerun = tmp_path / "retrieval_old.log"
    for path, score in [(full, 90), (smoke, 100), (rerun, 70)]:
        write_log(path, score)
    select(tmp_path, full)
    result = aggregate(tmp_path)
    assert len(result["sources"]) == 1
    assert [r["value"] for r in result["rows"]] == [90]
    write_log(full, 80)
    with pytest.raises(ValueError, match="changed since audit"):
        aggregate(tmp_path)


def test_manifest_rejects_duplicate_experiment_cells(tmp_path):
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    write_log(first)
    write_log(second)
    select(tmp_path, first, second)
    with pytest.raises(ValueError, match="duplicate result cells"):
        aggregate(tmp_path)


@pytest.mark.parametrize("backend,suite", [
    ("FovealLLM", "foveal_cached_generation"),
    ("ordinary-engine-index-disabled", "ordinary_engine_index_disabled"),
    (None, "unverified_generation"),
])
def test_serving_backend_is_not_inferred_from_checkpoint_label(backend, suite):
    result = {"model_config": {"is_foveal": True, "attn_type": "polar"}, "backend": backend,
              "results": {"2k": {"decode_tokens_per_s": 450}}}
    rows = _flatten("serving", result, Path("serving.log"))
    assert rows[0]["suite"] == suite
    assert rows[0]["backend"] == backend


def test_audited_snapshot_has_one_full_run_per_suite():
    folder = Path(__file__).resolve().parents[1] / "benchmarks/logs/foveal_cpt"
    result = aggregate(folder)
    assert len(result["sources"]) == 72
    assert len(result["rows"]) == 6336
    rows = [r for r in result["rows"] if r["model"] == "nope_kl" and r["benchmark"] == "retrieval"
            and r["suite"] == "synthetic" and r["length"] == "2k" and r["metric"] == "token_accuracy"]
    assert len(rows) == 6
    assert sum(r["value"] for r in rows) / 6 == pytest.approx(94.33333333)


def test_source_identity_survives_checkout_line_endings(tmp_path):
    path = tmp_path / "source.log"
    path.write_bytes(b"first\nsecond\n")
    before = source_hash(path)
    path.write_bytes(b"first\r\nsecond\r\n")
    assert source_hash(path) == before
