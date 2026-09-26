import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_belebele_diagonal_gmm import LANGUAGES, validate_config, validate_coverage
from run_mmlu_diagonal_gmm import model_source_paths


def test_joint_scope_and_sources_do_not_silently_use_one_language():
    cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/belebele_diagonal_gmm_20260926.json').read_text())
    validate_config(cfg)
    assert len(model_source_paths(cfg, cfg['models'][0])) == 3
    altered = copy.deepcopy(cfg)
    altered['models'][0]['source_paths'].pop()
    with pytest.raises(AssertionError):
        validate_config(altered)
    assert model_source_paths({'source_root': '/source'}, {'key': 'qwen2'}) == ['/source/qwen2']


def test_exact_six_cell_coverage_rejects_missing_duplicate_and_extra_rows():
    rows = [dict(model_alias=model, language=lang, start=0, samples=900,
                 sample_ids=[f'belebele_{code}_{i}' for i in range(900)])
            for model in ['qwen2', 'llama32'] for lang, code in LANGUAGES.items()]
    assert sum(validate_coverage(rows).values()) == 5400
    for mutate in ['missing', 'duplicate', 'extra']:
        altered = copy.deepcopy(rows)
        if mutate == 'missing':
            altered.pop()
        elif mutate == 'duplicate':
            altered[0]['sample_ids'][-1] = altered[0]['sample_ids'][0]
        else:
            altered.append(dict(model_alias='other', language='en', start=0, samples=1, sample_ids=['x']))
        with pytest.raises(ValueError):
            validate_coverage(altered)
