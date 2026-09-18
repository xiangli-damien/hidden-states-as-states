import numpy as np
import zarr
from hss.analysis.channel_data import token_samples, canonical_questions
from hss.analysis.channel_study import bh, split_rows, channel_stats
import pandas as pd


def test_token_alignment_and_short_response_missingness():
    values = np.arange(9 * 3, dtype=np.float32).reshape(9, 3)
    a = zarr.array(values, chunks=(4, 2))
    out = token_samples(a, np.array([0, 2, 9]), [1, 2, 4])
    np.testing.assert_array_equal(out[0, 0], values[0])
    np.testing.assert_array_equal(out[1, 0], values[2])
    np.testing.assert_array_equal(out[:, -1], values[[1, 8]])
    assert np.isnan(out[0, 2]).all()
    np.testing.assert_array_equal(out[1, 2], values[5])


def test_fdr_restores_order_and_controls_monotonicity():
    np.testing.assert_allclose(bh([.04,.001,.03,1]),[.053333333,.004,.053333333,1])


def test_prompt_deduplication_does_not_use_outcomes_or_subject():
    frame = pd.DataFrame({"sample_id":["subject_a_0","subject_b_9","subject_a_1"],
                          "prompt_text":["Which  option?\nA. 1","Which option? A. 1","Another question"],
                          "is_correct":[1,0,0]})
    result = canonical_questions(frame)
    assert result.canonical_question.tolist() == [True,False,True]
    assert result.prompt_sha256.iloc[0] == result.prompt_sha256.iloc[1]


def test_confirmation_and_nuisance_adjustment_remove_category_artifact():
    rng = np.random.default_rng(12)
    n = 2400
    category = rng.integers(0,2,n)
    y = (rng.random(n) < .15 + .7*category).astype(int)
    rows = pd.DataFrame({"y":y,"category":category.astype(str),"level":0,
                         "first_token":0,"n_prompt_tokens":30,"n_tokens":100,
                         "truncated":False,"parse_failed":False})
    x = rng.normal(size=(n,20))
    x[:,0] += 4*category
    x[:,1] += 2*y
    cfg = {"seed":7,"discovery_fraction":.4,"middle_rms_quantiles":[0,1],"massive_ratio":100}
    train,test = split_rows(rows,cfg)
    result = channel_stats(x,rows,train,test,cfg)
    assert abs(result.loc[0,"test_d"]) > 1
    assert abs(result.loc[0,"test_control_r"]) < .12
    assert result.loc[1,"test_control_r"] > .35
    assert result.loc[1,"test_control_p"] < 1e-15
