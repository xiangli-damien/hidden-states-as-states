from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_token_order import categorical_features,shuffle_questions,encode_training_tokens


def test_state_order_features_and_per_question_shuffles_preserve_counts():
    x=np.array([[0,1,1,2],[2,1,0,0]])
    counts=categorical_features(x,3,'occupancy').toarray()
    np.testing.assert_allclose(counts,[[.25,.5,.25],[.5,.25,.25]])
    ordered=categorical_features(x,3,'ordered').toarray().reshape(2,4,3)
    np.testing.assert_array_equal(ordered.argmax(-1),x)
    shuffled=shuffle_questions(x,['a','b'],42)
    np.testing.assert_array_equal(shuffled,shuffle_questions(x,['a','b'],42))
    np.testing.assert_allclose(categorical_features(shuffled,3,'occupancy').toarray(),counts)
    # A fixed global position permutation is only a column relabeling; question
    # order must not affect deterministic per-question shuffle assignment.
    np.testing.assert_array_equal(shuffle_questions(x[::-1],['b','a'],42),shuffled[::-1])


def test_transition_counts_are_adjacent_pairs_not_layer_matching():
    transition=categorical_features([[0,1,1,2]],3,'transitions').toarray()[0]
    expected=np.zeros(9);expected[[1,4,5]]=1/3
    np.testing.assert_allclose(transition,expected)


def test_token_vocabulary_fits_train_only_and_unknowns_get_reserved_slot():
    values=np.array([[7,8,7],[8,100,7]])
    encoded,vocab,known=encode_training_tokens(values,np.array([True,False]))
    np.testing.assert_array_equal(vocab,[7,8])
    np.testing.assert_array_equal(encoded,[[0,1,0],[1,2,0]])
    assert not known[1,1] and known.sum()==5
