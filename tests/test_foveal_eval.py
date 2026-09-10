"""Tests for Foveal CPT evaluation integration and inference engine adaptation."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from benchmarks.aggregate import _model_name
from benchmarks.model import (
    checkpoint_config,
    read_checkpoint_config,
    resolve_checkpoint,
    unsupported_features,
)
from benchmarks.scoring import DirectScorer, TokenRequest
from foveal_cpt.run_eval_pipeline import MODEL_SPECS, BenchmarkJob, _job_fingerprint


class FovealEvalTest(unittest.TestCase):
    def test_model_specs_all_12_present(self):
        self.assertEqual(len(MODEL_SPECS), 12)
        cores = {"polar", "nope", "rope"}
        variants = {"local", "lm_output", "kl", "lm_output_kl"}
        expected_keys = {f"{c}_{v}" for c in cores for v in variants}
        self.assertEqual(set(MODEL_SPECS.keys()), expected_keys)
        for key, spec in MODEL_SPECS.items():
            self.assertEqual(spec.key, key)
            self.assertIn(spec.core, cores)
            self.assertIn(spec.variant, variants)
            self.assertTrue(spec.subfolder.endswith("/cpt"))

    def test_resolve_checkpoint_with_latest_json(self):
        tmp_dir = Path("/tmp/opencode/test_resolve_ckpt")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "latest.json").write_text(
            json.dumps({"checkpoint": "cpt-step-001908.pt"}), encoding="utf-8"
        )
        (tmp_dir / "cpt-step-001908.pt").write_text("fake weights", encoding="utf-8")

        weights, ckpt_dir = resolve_checkpoint(str(tmp_dir))
        self.assertEqual(weights, str(tmp_dir / "cpt-step-001908.pt"))
        self.assertEqual(ckpt_dir, str(tmp_dir))

    def test_read_checkpoint_config_foveal_merge(self):
        tmp_dir = Path("/tmp/opencode/test_foveal_cfg")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        foveal_cfg = {
            "checkpoint": "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
            "adaptation_mode": "lm_output_kl",
            "sequence_length": 32768,
            "local_window": 512,
            "train_tokens": 1000000000,
        }
        (tmp_dir / "config.json").write_text(json.dumps(foveal_cfg), encoding="utf-8")
        (tmp_dir / "cpt-step-001908.pt").write_text("dummy", encoding="utf-8")

        cfg = read_checkpoint_config(str(tmp_dir))
        self.assertTrue(cfg.get("is_foveal"))
        self.assertEqual(cfg.get("adaptation_mode"), "lm_output_kl")
        self.assertEqual(cfg.get("attn_type"), "polar")
        self.assertEqual(cfg.get("local_window"), 512)
        self.assertEqual(cfg.get("attn_window"), 512)
        self.assertEqual(unsupported_features(cfg), [])

    def test_model_name_aggregate(self):
        cfg = {"is_foveal": True, "attn_type": "polar", "adaptation_mode": "lm_output_kl"}
        result = {"model_config": cfg}
        name = _model_name(result, Path("dummy.log"))
        self.assertEqual(name, "polar_lm_output_kl")

    def test_job_fingerprint_deterministic(self):
        job1 = BenchmarkJob(
            stage="smoke",
            benchmark="retrieval",
            suite="synthetic",
            model="polar_local",
            command_args=("--lengths", "2k"),
        )
        job2 = BenchmarkJob(
            stage="smoke",
            benchmark="retrieval",
            suite="synthetic",
            model="polar_local",
            command_args=("--lengths", "2k"),
        )
        self.assertEqual(_job_fingerprint(job1), _job_fingerprint(job2))

    def test_direct_scorer_padding_arbitrary_length(self):
        class MockFovealModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(100, 64)
                self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
                self.norm = torch.nn.Identity()
                self.proj = torch.nn.Linear(64, 100, bias=False)

            def forward(self, x):
                return x

        scorer = object.__new__(DirectScorer)
        scorer.model = MockFovealModel()
        scorer.device = torch.device("cpu")
        scorer.max_length = 2048
        scorer.tokenizer = type("Tokenizer", (), {"eos_token_id": 0})()
        scorer.cfg = {"is_foveal": True, "attn_type": "polar"}

        # Forward sequence of length 75 (not divisible by 64)
        input_ids = torch.randint(0, 100, (1, 75))
        hidden = scorer._forward_hidden(input_ids)
        self.assertEqual(hidden.shape, (1, 75, 64))

        # Forward sequence of length 64 (divisible by 64)
        input_ids_64 = torch.randint(0, 100, (1, 64))
        hidden_64 = scorer._forward_hidden(input_ids_64)
        self.assertEqual(hidden_64.shape, (1, 64, 64))

    def test_pipeline_stages_resolution(self):
        from foveal_cpt.run_eval_pipeline import _resolve_stages, _build_jobs
        import argparse

        # "all" should expand to all 5 core benchmark stages
        stages = _resolve_stages(["all"])
        self.assertEqual(stages, {"base", "retrieval", "longdoc", "babilong_pipeline", "serving"})

        # Multiple explicit stages
        multi = _resolve_stages(["base", "retrieval"])
        self.assertEqual(multi, {"base", "retrieval"})


if __name__ == "__main__":
    unittest.main()
