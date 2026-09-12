"""Audit acceptance must reject stale, nonfinite, or structurally invalid evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import summarize_foveal_audit as audit


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'CORES', ('polar',))
    monkeypatch.setattr(audit, 'MODES', ('kl',))
    monkeypatch.setattr(audit, 'BF16_CASES', {'generated_pages': 2})
    monkeypatch.setattr(audit, 'FP32_CASES', {'fp32': 2})
    zero = dict(max_abs_logit_error=0., rms_logit_error=0., relative_l2_logit_error=0.,
                within_tolerance=True, greedy_agreement=True)
    error = dict(max_abs_logit_error=.2, rms_logit_error=.02, relative_l2_logit_error=.01,
                 within_tolerance=False, greedy_agreement=False)
    controls = {name: dict(error) for name in audit.CONTROLS}
    rows = [dict(zero, **{name: dict(zero) for name in audit.CONTROLS}), dict(error, **controls)]
    case = dict(comparisons=rows, final_routes={'2': {'cached': [1], 'reference': [1]}},
                final_state_relative_l2_errors={name: {'2': .01} for name in
                                               ('mem_states', 'k_cache', 'v_cache', 'k_pages', 'v_pages')})
    report = dict(decoder_sha256=hashlib.sha256(Path('baseline_inference/foveal_engine.py').read_bytes()).hexdigest(),
                  passed=False, cases=[case])
    (tmp_path/'polar_kl_generated_pages.json').write_text(json.dumps(report))
    fp = copy.deepcopy(report)
    fp['passed'] = True
    fp['cases'][0]['comparisons'] = [dict(zero), dict(zero)]
    (tmp_path/'polar_kl_fp32.json').write_text(json.dumps(fp))
    return tmp_path


def test_numerical_acceptance_preserves_strict_bf16_failure(evidence):
    result = audit.summarize(evidence)['models']['polar_kl']
    assert result['serving_eligible']
    assert result['bf16']['strict_failures'] == 1
    assert result['bf16']['greedy_disagreements'] == 1


@pytest.mark.parametrize('failure', ['stale', 'nonfinite', 'fp32', 'envelope', 'generated_pages'])
def test_bad_evidence_never_allows_serving(evidence, failure):
    path = evidence / ('polar_kl_fp32.json' if failure == 'fp32' else 'polar_kl_generated_pages.json')
    report = json.loads(path.read_text())
    if failure == 'stale':
        report['decoder_sha256'] = 'old source'
    elif failure == 'nonfinite':
        report['cases'][0]['comparisons'][1]['rms_logit_error'] = None
    elif failure == 'fp32':
        report['passed'] = False
    elif failure == 'envelope':
        report['cases'][0]['comparisons'][1]['rms_logit_error'] = .5
    else:
        report['cases'][0]['final_routes']['2']['cached'] = [0]
    path.write_text(json.dumps(report))
    assert not audit.summarize(evidence)['models']['polar_kl']['serving_eligible']
