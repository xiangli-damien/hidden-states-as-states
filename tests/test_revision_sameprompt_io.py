import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(scripts))
from revision_sameprompt_io import confidence, summaries, valid_prefix, verify_record_files
from revision_common import sha


def test_boundaries_do_not_borrow_prompt_or_include_eos():
    assert valid_prefix(list(range(20)), 16, {99})
    assert not valid_prefix(list(range(16)), 16, {99})
    assert not valid_prefix(list(range(20)), 16, {10})
    assert valid_prefix(list(range(20)), 16, {19})


def test_means_use_actual_generated_positions_and_keep_layer_axis():
    raw = np.arange(2*64*3, dtype=np.float32).reshape(2, 64, 3); post = raw[-1]*2
    got = summaries(raw, post, 16)
    np.testing.assert_array_equal(got['current'], raw[:, 63])
    np.testing.assert_array_equal(got['mean_window'], raw[:, 48:64].mean(1))
    np.testing.assert_array_equal(got['mean_all'], raw.mean(1))
    assert not np.array_equal(got['mean_window'], got['mean_all'])
    with pytest.raises(ValueError): summaries(raw.reshape(128, 3), post, 16)


def test_raw_confidence_is_invariant_to_logit_offset():
    logits = np.array([2., -1., .2]); lp = logits-np.log(np.exp(logits).sum())
    original = confidence(logits, lp); shifted = confidence(logits+1000, lp)
    assert original['entropy'] == shifted['entropy']
    np.testing.assert_allclose(original['margin'], shifted['margin'])
    with pytest.raises(ValueError): confidence(logits, lp+1)


def test_resume_refuses_corrupt_or_outside_artifacts(tmp_path):
    p = tmp_path/'saved.bin'; p.write_bytes(b'valid')
    record = {'files': {'saved.bin': sha(p)}}; verify_record_files(tmp_path, record)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError): verify_record_files(tmp_path, record)
    with pytest.raises(ValueError): verify_record_files(tmp_path, {'files': {'../outside': 'unused'}})
