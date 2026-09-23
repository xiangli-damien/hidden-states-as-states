"""Synthetic, low-dimensional end-to-end fixture; never an LLM research result."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from revision_common import write_json, write_npz, sha
from fit_revision_sameprompt import run as fit
from audit_revision_sameprompt_readouts import run as audit


def test_complete_synthetic_readout_and_independent_audit(tmp_path):
    rng = np.random.default_rng(8); root = tmp_path/'synthetic_only'; root.mkdir()
    counts = {'train': 96, 'tuning': 32, 'calibration': 32, 'test': 64}
    cfg = {'role_counts': counts, 'layers': [14, 28], 'prefixes': [16, 64],
           'window': 16, 'readout_C_grid': [.001, .01, .1], 'max_iter': 200}
    items = []; records = {}; question = 0
    for role, count in counts.items():
        for _ in range(count):
            sid = f'synthetic_{question:04}'; group = f'group_{question:04}'
            item = {'sample_id': sid, 'question_group': group, 'role': role, 'prompt_ids': [1, 2],
                    'category': str(question % 3), 'level': str(question % 5)}
            items.append(item); question += 1
            prompt = rng.normal(size=(2, 1, 4)).astype(np.float32)
            promptfile = f'prompts/{sid}.npz'
            write_npz(root/'full'/promptfile, raw=prompt, current=prompt[:, -1])
            for repeat in range(4):
                tid = f'{sid}_r{repeat}'; failure = (question+repeat) % 2
                response = rng.normal(size=(2, 64, 4)).astype(np.float32)+.2*failure
                record = {'sample_id': sid, 'question_group': group, 'role': role,
                    'trajectory': {'trajectory_id': tid}, 'correct': not failure, 'length': 100,
                    'finish_reason': 'eos', 'prompt_file': promptfile,
                    'files': {promptfile: sha(root/'full'/promptfile)}, 'prefixes': {}}
                for prefix in [16, 64]:
                    raw = response[:, :prefix]
                    logits = rng.normal(size=5).astype(np.float32)
                    lp = (logits.astype(float)-np.log(np.exp(logits.astype(float)).sum())).astype(np.float32)
                    path = f'features/{tid}_p{prefix}.npz'
                    write_npz(root/'full'/path, raw=raw, current=raw[:, -1],
                        mean_window=raw[:, -16:].mean(1, dtype=np.float64).astype(np.float32),
                        mean_all=raw.mean(1, dtype=np.float64).astype(np.float32), logits=logits, logprobs=lp)
                    order = np.sort(logits.astype(float))
                    record['prefixes'][str(prefix)] = {'valid': True, 'file': path,
                        'entropy': float(-(np.exp(lp.astype(float))*lp).sum()), 'margin': float(order[-1]-order[-2])}
                    record['files'][path] = sha(root/'full'/path)
                name = f'{tid}.json'; write_json(root/'full/samples'/name, record)
                records[name] = sha(root/'full/samples'/name)
    write_json(root/'inputs.json', items)
    write_json(root/'plan.json', {'config': cfg, 'files': {}, 'inputs_sha256': sha(root/'inputs.json'), 'synthetic_test_only': True})
    write_json(root/'preparation.json', {'complete': True, 'plan_sha256': sha(root/'plan.json')})
    write_json(root/'full/_SUCCESS.json', {'complete': True, 'records': records, 'synthetic_test_only': True})
    write_json(root/'full/audit.json', {'complete': True, 'stage_receipt_sha256': sha(root/'full/_SUCCESS.json'),
        'synthetic_fixture_envelope_not_an_llm_audit': True})
    fit(root); audit(root)
    result = json.loads((root/'readouts/audit.json').read_text())
    assert result['complete'] and result['methods'] == 10 and result['candidates'] == 30
    assert result['worst_prediction_error'] < 1e-10
