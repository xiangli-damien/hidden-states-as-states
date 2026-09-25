from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from delta_cluster_lookup_common import token_counts, fit_error_lookup, lookup_sequence


def test_raw_token_weighting_train_only_and_no_smoothing():
    # One 9-token wrong answer and one 1-token correct answer: 90%, not 50%.
    counts = np.array([[[9, 0]], [[1, 0]], [[1000, 1000]]])
    a = fit_error_lookup(counts, [1, 0, 0], [True, True, False])
    b = fit_error_lookup(counts, [1, 0, 1], [True, True, False])
    np.testing.assert_array_equal(a['q'], b['q'])
    assert a['q'][0, 0] == .9
    assert a['fallback'][0, 1] and a['q'][0, 1] == .9


def test_layer_isolation_and_prefix_causality():
    q = np.array([[.1, .9], [.8, .2]])
    sequence = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    current, mean = lookup_sequence(sequence, q)
    np.testing.assert_allclose(current[:2], [[.1, .8], [.9, .8]])
    np.testing.assert_allclose(mean[1], [.5, .8])
    altered = sequence.copy(); altered[2:] = 1 - altered[2:]
    a, b = lookup_sequence(altered, q)
    np.testing.assert_array_equal(a[:2], current[:2])
    np.testing.assert_array_equal(b[:2], mean[:2])
    altered = sequence.copy()
    altered[:, 1] = 1 - altered[:, 1]
    a, b = lookup_sequence(altered, q)
    np.testing.assert_array_equal(a[:, 0], current[:, 0])
    np.testing.assert_array_equal(b[:, 0], mean[:, 0])


def test_counts_and_id_renaming():
    seq = np.array([[0, 1], [1, 1], [1, 0]])
    np.testing.assert_array_equal(token_counts(seq, [2, 2]), [[1, 2], [1, 2]])
    q = np.array([[.2, .7], [.3, .9]])
    a, b = lookup_sequence(seq, q)
    c, d = lookup_sequence(1 - seq, q[:, ::-1])
    np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
    with pytest.raises(ValueError): token_counts(np.array([[-1, 0]]), [2, 2])


def test_synthetic_end_to_end_lookup_metrics_and_report(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import run_delta_cluster_lookup as runner
    from revision_common import write_json
    n = 12; layers = [1, 7, 14, 21, 28]
    rng = np.random.default_rng(31)
    seq = [rng.integers(0, 32, size=(10+i, 5), dtype=np.int16) for i in range(n)]
    rows = pd.DataFrame({'sample_id': [f's{i}' for i in range(n)],
                         'split': ['train']*6+['validation']*2+['test']*4,
                         'n_tokens': [len(s) for s in seq], 'label': [0, 1]*6,
                         'prompt_text': ['synthetic question']*n, 'response_text': ['synthetic answer']*n})
    src = tmp_path/'source'; previous=tmp_path/'previous'; previous.mkdir()
    token_meta={f's{i}': {'row_index': i, 'response_ids': list(range(len(s)))} for i,s in enumerate(seq)}
    write_json(previous/'tokens.json', token_meta)
    for layer in layers:
        folder=src/'fits'/f'token_delta_L{layer:02d}';folder.mkdir(parents=True)
        (folder/'selected.joblib').write_text('synthetic fixture, not a trained model')
    tokenizer=tmp_path/'tokenizer.json';tokenizer.write_text('{}')
    cfg={'output': str(tmp_path/'result'), 'source': str(src), 'previous': str(previous),
         'layers': layers, 'prefixes': [1,4,8], 'primary_prefix': 4,
         'seed': 9, 'bootstrap': 20, 'case_count': 2, 'tokenizer': str(tokenizer), 'tokenizer_python': sys.executable}
    cp=tmp_path/'config.json';write_json(cp,cfg)
    shard={'name': 'fixture', 'question_index': np.arange(n), 'token_ptr': np.r_[0,np.cumsum(rows.n_tokens)]}
    monkeypatch.setattr(runner,'load_data',lambda _: (rows,seq,[shard],[32]*5,[]))
    original=runner.subprocess.check_output
    def decode_stub(args, **kwargs):
        if len(args)>2 and args[1]=='-c':
            return json.dumps({k:[str(x) for x in v] for k,v in json.loads(kwargs['input']).items()})
        return original(args,**kwargs)
    monkeypatch.setattr(runner.subprocess,'check_output',decode_stub)
    runner.run(cfg,cp)
    out=tmp_path/'result'
    assert json.loads((out/'_SUCCESS.json').read_text())['metrics']==30
    assert json.loads((out/'audit.json').read_text())['complete']
    with np.load(out/'lookup.npz') as z:
        assert int((z['wrong']+z['correct']).sum()) == sum(map(len,seq[:6]))*5
    metrics=pd.read_csv(out/'metrics.csv')
    assert len(metrics)==30 and metrics.auroc.between(0,1).all()
    assert (out/'report/index.html').exists()
    assert json.loads((out/'summary.json').read_text())['primary_prefix']==4
