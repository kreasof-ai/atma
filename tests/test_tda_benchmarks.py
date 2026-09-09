"""External checkpoints must use their own factory and retain strict key checks."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.scoring import DirectScorer


@pytest.mark.parametrize('arch', ['tda_hybrid', 'mamba3_native', 'gdn2_native'])
def test_external_scorer_routes_to_pinned_factory(monkeypatch, arch):
    import sys
    from external_baselines import verify_checkpoint
    from supplementary.robustness import run_worker

    verify = Mock()
    add_sources = Mock()
    model = Mock()
    model.blocks = []
    model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
    factory = Mock(return_value=model)
    monkeypatch.setattr(run_worker, '_verify_external_dependencies', verify)
    monkeypatch.setattr(verify_checkpoint, '_add_pinned_sources', add_sources)
    monkeypatch.setitem(sys.modules, 'external_baselines.model', SimpleNamespace(create_model=factory))
    scorer = DirectScorer.__new__(DirectScorer)
    scorer.cfg = {'arch_type': arch, 'baseline_family': 'external'}
    scorer.weights_path = 'weights.pt'
    scorer.device = 'cpu'
    scorer.gamma_clamp = None
    torch = SimpleNamespace(load=Mock(return_value={'model': {'_orig_mod.weight': 1}}))
    assert scorer._load_model(torch) is model
    verify.assert_called_once_with(scorer.cfg)
    add_sources.assert_called_once()
    factory.assert_called_once_with(scorer.cfg)
    model.load_state_dict.assert_called_once_with({'weight': 1}, strict=False)
    model.to.assert_called_once_with('cpu')
    model.eval.assert_called_once()
    model.load_state_dict.return_value.missing_keys = ['missing.weight']
    with pytest.raises(RuntimeError, match='checkpoint layout mismatch'):
        scorer._load_model(torch)


def test_direct_serving_counts_first_token_as_prefill_and_ignores_eos():
    import torch
    from benchmarks.direct_serving import measure_sequence

    lengths = []
    def forward(inputs):
        lengths.append(inputs.shape[1])
        return torch.zeros((*inputs.shape, 2))
    scorer = SimpleNamespace(
        device=torch.device('cpu'), _forward_hidden=forward,
        model=SimpleNamespace(norm=lambda x: x, proj=lambda x: x),
    )
    result = measure_sequence(scorer, [1, 2, 3], 3)
    assert lengths == [3, 4, 5]
    assert result['prefill_tokens'] == 3
    assert result['decode_tokens'] == 2
    assert result['generated_tokens'] == 3
    assert result['prefill_time_s'] > 0
    assert result['decode_time_s'] > 0


def test_direct_serving_oom_is_recorded_and_next_length_runs(monkeypatch):
    import torch
    from benchmarks import direct_serving, scoring, serving
    from benchmarks import model as model_module

    closed = []
    class Scorer:
        def __init__(self, *args, **kwargs):
            self.device = torch.device('cpu')
            self.tokenizer = None
        def close(self):
            closed.append(True)
    def measure(scorer, prompt, decode_tokens):
        if len(prompt) == 4:
            raise torch.cuda.OutOfMemoryError('synthetic OOM')
        return dict(prefill_tokens=len(prompt), decode_tokens=1, generated_tokens=2,
                    prefill_time_s=2., decode_time_s=1.)
    monkeypatch.setattr(scoring, 'DirectScorer', Scorer)
    monkeypatch.setattr(serving, '_prompt_ids', lambda tok, n: [0] * n)
    monkeypatch.setattr(serving, '_clear_cuda', lambda: None)
    monkeypatch.setattr(direct_serving, 'measure_sequence', measure)
    monkeypatch.setattr(model_module, 'read_checkpoint_config', lambda p: {'arch_type': 'tda_hybrid'})
    result = direct_serving.run_direct_serving('checkpoint', ['4', '2'], decode_tokens=2,
                                               samples=2, log_fn=lambda s: None)
    assert result['results']['4']['oom']
    assert not result['results']['2']['oom']
    assert result['results']['2']['prefill_tokens'] == 4
    assert result['results']['2']['time_to_first_token_s'] == 2
    assert result['max_successful_context_tokens'] == 2
    assert len(closed) == 2


def test_aggregate_keeps_recomputed_serving_separate():
    from pathlib import Path
    from benchmarks.aggregate import _flatten
    result = {'backend': 'direct-full-recompute', 'model_config': {'arch_type': 'tda_hybrid'},
              'results': {'2k': {'time_to_first_token_s': 1., 'decode_tokens_per_s': 2.,
                                  'samples': 1, 'oom': False}},
              'max_successful_context_tokens': 2048}
    rows = _flatten('serving', result, Path('tda.log'))
    assert rows and all(row['suite'] == 'direct_full_recompute' for row in rows)
    assert any(row['metric'] == 'time_to_first_token_s' for row in rows)


def test_tda_paged_serving_fails_with_actionable_error():
    from benchmarks.model import checkpoint_config
    with pytest.raises(NotImplementedError, match='--serving_backend direct'):
        checkpoint_config({'arch_type': 'tda_hybrid'})


def test_tda_pipeline_resumes_and_rejects_changed_checkpoint(tmp_path, monkeypatch):
    import json
    import sys
    from supplementary.robustness import benchmark_tda as runner
    from benchmarks import babilong
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'arch_type': 'tda_hybrid'}))
    (model / 'weights.pt').write_bytes(b'checkpoint')
    output = tmp_path / 'output'
    calls = []
    monkeypatch.setattr(babilong, '_load_official_prompts', lambda: None)
    def jobs(model, output, adapted):
        return [('base', ['fake'], output / 'results/base.log')]
    def run(command, console, env):
        calls.append(command)
        log = output / 'results/base.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('===BASE_RESULTS_JSON===\n{}\n===END===\n')
        return 0
    monkeypatch.setattr(runner, 'jobs', jobs)
    monkeypatch.setattr(runner, '_run_with_console_log', run)
    monkeypatch.setattr(sys, 'argv', ['runner', '--model', str(model), '--output-dir', str(output)])
    runner.main()
    runner.main()
    assert len(calls) == 1
    assert (output / 'benchmark_matrix.json').exists()
    (model / 'weights.pt').write_bytes(b'changed')
    with pytest.raises(SystemExit, match='Checkpoint/code/plan changed'):
        runner.main()


def test_tda_pipeline_gate_rejects_oom(tmp_path):
    import json
    from supplementary.robustness.benchmark_tda import gate_passed
    log = tmp_path / 'gate.log'
    result = {'counts': {'qa1': {'256k': 1}}, 'oom_cells': []}
    def write():
        log.write_text('===BABILONG_RESULTS_JSON===\n' + json.dumps(result) + '\n===END===\n')
    write()
    assert gate_passed(log)
    result['oom_cells'] = [{'task': 'qa1', 'length': '256k'}]
    write()
    assert not gate_passed(log)
