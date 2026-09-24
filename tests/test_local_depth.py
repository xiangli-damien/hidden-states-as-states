import numpy as np
import pytest

from hss.analysis.local_depth import (balanced_positions, segment_means,
                                      sequence_features, SequenceBayes,
                                      pack_bf16_exact, unpack_bf16)


def test_cache_is_lossless_and_refuses_implicit_precision_change():
    x = np.array([0, -0., 1.25, -61440., 2**20], dtype=np.float32)
    np.testing.assert_array_equal(unpack_bf16(pack_bf16_exact(x)).view(np.uint32), x.view(np.uint32))
    with pytest.raises(ValueError): pack_bf16_exact(np.array([1.000001],np.float32))


def test_sentence_delta_and_weighted_average_recover_whole_answer():
    rng = np.random.default_rng(41)
    h = rng.normal(size=(17, 4, 9)).astype(np.float32)
    ends = [3, 7, 17]
    delta = h[:, 1:] - h[:, :-1]
    segmented = segment_means(delta, ends)
    means = segment_means(h, ends)
    np.testing.assert_allclose(segmented, means[:, 1:]-means[:, :-1], atol=3e-7)
    np.testing.assert_allclose(np.average(segmented, axis=0, weights=np.diff([0]+ends)),
                               delta.mean(0), atol=3e-7)
    with pytest.raises(ValueError): segment_means(delta, [3, 7])


def test_every_question_has_equal_sampling_mass():
    for n in [1, 3, 100]:
        a = balanced_positions(n, 'q', 4, 12, 'fit')
        assert len(a) == 4 and np.all((a >= 0) & (a < n))
        np.testing.assert_array_equal(a, balanced_positions(n, 'q', 4, 12, 'fit'))


def test_shuffling_preserves_occupancy_but_not_edges_and_no_boundary_crossing():
    sequences = [np.array([[0], [0], [1], [1]]), np.array([[1]])]
    occ, edge = sequence_features(sequences, [2], ['a', 'b'], 13)
    np.testing.assert_allclose(occ, [[.5,.5], [0,1]])
    np.testing.assert_allclose(edge, [[1/3,1/3,0,1/3], [0,0,0,0]])
    changed = False
    for seed in range(10):
        oo, ee = sequence_features(sequences, [2], ['a', 'b'], seed, True)
        np.testing.assert_array_equal(oo, occ)
        np.testing.assert_array_equal(ee[1], 0)
        changed |= not np.allclose(ee[0], edge[0])
    assert changed


def test_order_signal_with_identical_occupancy():
    a = np.array([[0],[0],[1],[1]])
    b = np.array([[0],[1],[0],[1]])
    seq = [a]*20 + [b]*20
    occ, edges = sequence_features(seq, [2], [str(i) for i in range(40)], 1)
    y = np.r_[np.zeros(20,int),np.ones(20,int)]
    nb = SequenceBayes([2]).fit(occ,y)
    np.testing.assert_allclose(nb.decision_function(occ), 0)
    xx = np.column_stack([occ,edges])
    model = SequenceBayes([2], transitions=True).fit(xx,y)
    score = model.decision_function(xx)
    assert score[20:].min() > score[:20].max()
