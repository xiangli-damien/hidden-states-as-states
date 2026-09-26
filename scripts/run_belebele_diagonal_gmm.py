"""Two-model BELEBELE response-mean maps, reusing the audited MMLU fitter.

This fits one joint English/German/Chinese map per model. Correctness and
language remain metadata: neither is used in fitting or selecting K.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from importlib.metadata import version

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_mmlu_diagonal_gmm as common
from audit_mmlu_diagonal_results import audit
from revision_common import freeze, sha, status, write_json

LANGUAGES = {"en": "eng_Latn", "de": "deu_Latn", "zh": "zho_Hans"}
MODELS = {"qwen2": ("Qwen/Qwen2-7B-Instruct", 28, 3584),
          "llama32": ("meta-llama/Llama-3.2-1B-Instruct", 16, 2048)}


def validate_config(cfg):
    assert cfg['dataset_id'] == 'belebele' and cfg['samples'] == 2700
    assert cfg['languages'] == LANGUAGES and cfg['language_mode'] == 'joint'
    assert cfg['representation'] == 'mean' and cfg['normalization'] is False
    assert cfg['covariance_type'] == 'diag' and cfg['include_pre_final'] is True
    assert cfg['icl_tolerances'] == [0.0, 0.02] and cfg['primary_tolerance'] == .02
    assert cfg['n_init'] == 3 and len(cfg['models']) == len(MODELS)
    assert {m['key'] for m in cfg['models']} == set(MODELS)
    for m in cfg['models']:
        assert (m['identifier'], m['last_layer'], m['dimension']) == MODELS[m['key']]
        expected = [str(Path(cfg['source_root'])/m['key']/f'belebele_{l}') for l in LANGUAGES]
        assert common.model_source_paths(cfg, m) == expected


def validate_coverage(rows):
    coverage = {}
    for key in MODELS:
        for lang, code in LANGUAGES.items():
            part = sorted((r for r in rows if r['model_alias'] == key and r['language'] == lang),
                          key=lambda r: r['start'])
            ids = [sid for r in part for sid in r['sample_ids']]
            if ids != [f'belebele_{code}_{i}' for i in range(900)]:
                raise ValueError(f'Missing, repeated or reordered source coverage: {key}/{lang}')
            if sum(r['samples'] for r in part) != 900:
                raise ValueError(f'Inconsistent sample counts: {key}/{lang}')
            coverage[f'{key}/{lang}'] = len(ids)
    if sum(r['samples'] for r in rows) != 5400:
        raise ValueError('Unexpected source rows outside the six authorized cells')
    return coverage


def verify_collection(cfg):
    root = Path(cfg['collection_root'])
    source = json.loads((root/'job_status.json').read_text())
    end = json.loads((root/'_SUCCESS').read_text())
    assert source['status'] == 'complete' and source['completed_samples'] == end['samples'] == 5400
    assert sha(root/'job_status.json') == end['status_sha256']
    assert all(v == 0 for v in source['exit_codes'].values())
    coverage = validate_coverage(source['rows'])
    inputs = []
    for row in source['rows']:
        p = Path(row['run_path'])
        assert p.parent == Path(cfg['source_root'])/row['model_alias']/f"belebele_{row['language']}"
        record = json.loads((p/'_SHARD.json').read_text())
        receipt = json.loads((p/'_COPY_VERIFIED.json').read_text())
        assert record['passed'] and record['fingerprint'] == source['fingerprint']
        assert record['sample_ids'] == row['sample_ids']
        assert record['verification']['fixed_shape_exact_replay_and_causality']
        assert record['evaluation']['n_error'] == 0 and record['evaluation']['n_evaluated'] == row['samples']
        checked = {}
        for name, info in receipt['files'].items():
            # Rehash every input used by this mean-only fit, plus publication metadata.
            if name in ['manifest.json', 'data.parquet', '_SUCCESS', '_SHARD.json',
                        'labels/correctness.parquet', 'tensors.zarr/.zmetadata'] or name.startswith((
                            'tensors.zarr/hidden_states/mean/', 'tensors.zarr/final_norm/pre/mean/',
                            'tensors.zarr/tokens/sample_ptr/')):
                assert sha(p/name) == info['sha256'], str(p/name)
                checked[name] = info['sha256']
        manifest = json.loads((p/'manifest.json').read_text())
        assert manifest['dataset']['name'] == 'belebele'
        assert manifest['model']['identifier'] == MODELS[row['model_alias']][0]
        frame = pd.read_parquet(p/'data.parquet')
        labels = pd.read_parquet(p/'labels/correctness.parquet')
        assert frame.sample_id.tolist() == row['sample_ids'] and len(labels) == row['samples']
        assert not labels.sample_idx.duplicated().any() and labels.is_correct.isin([True, False, 0, 1]).all()
        inputs.append(dict(path=str(p), receipt_sha256=sha(p/'_COPY_VERIFIED.json'), input_hashes=checked))
    return dict(coverage=coverage, completed_before_fitting=True, inputs=inputs,
                collection_status_sha256=sha(root/'job_status.json'),
                collection_success_sha256=sha(root/'_SUCCESS'), plan_sha256=sha(root/'job_plan.json'))


def freeze_protocol(root, cfg):
    repo = Path(__file__).resolve().parents[1]
    files = [Path(__file__), Path(common.__file__), repo/'scripts/audit_mmlu_diagonal_results.py',
             repo/'scripts/revision_common.py', repo/'src/hss/cluster/gmm_resident.py',
             repo/'src/hss/cluster/gmm.py', repo/'src/hss/data/openact.py',
             repo/'src/hss/data/spec.py', repo/'src/hss/align.py', repo/'configs/base.toml']
    identity = dict(config=cfg, source_sha256={str(p): sha(p) for p in files},
                    environment={k: version(k) for k in ['numpy', 'torch', 'scikit-learn', 'zarr']},
                    fit_scope='all2700 per model, three languages jointly; descriptive unlabeled geometry',
                    selection='ICL=BIC+2*posterior entropy; converged only; smallest K in 0/2% relative tolerance',
                    search='K1..16 then step4 to80, boundary extension96..160, two integer refinement rounds',
                    search_limit='Optimal among evaluated candidates, not exhaustive global optimum',
                    precision='float64; translation for numerical stability only; no feature normalization',
                    assignments='GMM posterior MAP; nearest-center assignments also retained',
                    downstream='No correctness predictor, held-out AUROC, MFA or steering')
    p = root/'protocol.json'
    if p.exists():
        assert json.loads(p.read_text())['identity'] == identity, 'Frozen source/config changed'
    else:
        write_json(p, dict(identity=identity, frozen_unix=time.time(),
                          git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()))


def validate_cache(data, model):
    assert data.n_items() == 2700 and data.state_dim() == model['dimension']
    expected = {f'belebele_{code}_{i}' for code in LANGUAGES.values() for i in range(900)}
    assert set(data.meta.sample_id) == expected and data.meta.sample_id.nunique() == 2700
    assert data.meta.language.value_counts().to_dict() == {'en': 900, 'de': 900, 'zh': 900}
    assert data.meta.label.isin([0, 1]).all()


def run(path):
    cfg = json.loads(Path(path).read_text()); validate_config(cfg)
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    with (root/'study.lock').open('a') as handle, threadpool_limits(cfg['threads']):
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'COMPLETE.json').exists():
            raise RuntimeError('Already complete: do not launch another fit')
        freeze_protocol(root, cfg)
        status(root, 'queue', state='auditing_collection')
        freeze(root/'collection_receipt.json', verify_collection(cfg))
        snapshots = {}
        for model in cfg['models']:
            snapshots[model['key']] = common.prepare_model(root, cfg, model)
            for data in snapshots[model['key']].values():
                validate_cache(data, model)
            post, pre = snapshots[model['key']]['post'], snapshots[model['key']]['pre_final']
            assert post.meta.sample_id.tolist() == pre.meta.sample_id.tolist()
        cancelled = json.loads(Path(cfg['cancelled_reliability_status']).read_text())
        assert cancelled['status'] == 'cancelled_by_user', 'Reliability must remain cancelled'
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError(f'GPU still used by another job: {active}')
        with Path('/tmp/hss-gpu-cuda_0.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            import torch
            torch.set_num_threads(cfg['threads'])
            common.gpu_preflight(root)
            for model in cfg['models']:
                if not (root/model['key']/'COMPLETE.json').exists():
                    common.run_model(root, cfg, model, snapshots[model['key']])
                status(root, 'queue', state='auditing_model', model=model['key'])
                audit(root, model['key'])
        receipts = {m['key']: sha(root/m['key']/'independent_audit.json') for m in cfg['models']}
        write_json(root/'COMPLETE.json', dict(completed_unix=time.time(), models=list(receipts),
                   independent_audits=receipts, protocol_sha256=sha(root/'protocol.json')))
        status(root, 'queue', state='complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException:
        cfg = json.loads(Path(args.config).read_text()); root = Path(cfg['output'])
        write_json(root/'failure.json', dict(at=time.time(), traceback=traceback.format_exc()))
        status(root, 'queue', state='failed')
        raise
