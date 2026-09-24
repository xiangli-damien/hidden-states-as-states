"""Read three preselected native OpenAct sequences; no inference or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = Path('/lambda/nfs/dami/hss/tokenmean-replacement-20260924')
    cases = [r for r in json.loads((study/'cases.json').read_text()) if r['dataset']=='math'][:3]
    assert [c['sample_id'] for c in cases] == ['math_3992', 'math_2916', 'math_2668']
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct',
        revision='f2826a00ceef68f0f2b946d945ecc0477ce4450c', local_files_only=True)
    base = Path('/lambda/nfs/dami/openact/runs/math_full_20260916/qwen2')
    cache = Path('/home/ubuntu/hss-cache/data/473f2199a8742bdf18996361')
    meta = pd.read_parquet(cache/'rows.parquet')
    old_means = np.load(cache/'layer_14.npy', mmap_mode='r')
    rows = []
    for case in cases:
        sid = case['sample_id']; number = int(sid.split('_')[-1]); start = number//100*100
        shard = base/f'shard_{start:05d}_{start+100:05d}'
        table = pd.read_parquet(shard/'data.parquet')
        r = table[table.sample_id.eq(sid)].iloc[0]; i = int(r.sample_idx)
        z = zarr.open_group(str(shard/'tensors.zarr'), mode='r')
        a,b = map(int, z['tokens/sample_ptr'][i:i+2])
        ids = np.asarray(z['tokens/ids'][a:b])
        np.testing.assert_array_equal(ids, case['response_ids'])
        assert json.loads(r.prompt_token_ids_json)==case['prompt_ids']
        x = np.asarray(z['hidden_states/per_token'][a:b,14,:], dtype=np.float32)
        prompt = np.asarray(z['hidden_states/prompt_last'][i,14,:], dtype=np.float32)
        stored = np.asarray(z['hidden_states/mean'][i,14,:], dtype=np.float32)
        assert x.shape==(len(ids),3584) and np.isfinite(x).all()
        recomputed = x.astype(np.float64).mean(0)
        np.testing.assert_allclose(stored, recomputed, atol=2e-5, rtol=2e-6)
        idx = np.flatnonzero(meta.sample_id.eq(sid))[0]
        np.testing.assert_array_equal(stored, old_means[idx])
        output = args.output/(sid+'.npz')
        np.savez_compressed(output, x=x, prompt_last=prompt, full_mean_stored=stored, token_ids=ids)
        rows.append({'sample_id':sid, 'problem':r.problem, 'response_text':r.response_text,
            'ground_truth':str(r.ground_truth), 'tokens':[tokenizer.decode([int(t)],clean_up_tokenization_spaces=False) for t in ids],
            'n_tokens':len(ids), 'category':r.category, 'array_file':output.name,
            'array_sha256':digest(output), 'shard':str(shard),
            'metadata_sha256':digest(shard/'data.parquet'), 'manifest_sha256':digest(shard/'manifest.json'),
            'mean_cache_exact':True, 'mean_recompute_max_abs':float(np.max(abs(stored-recomputed)))})
        print(json.dumps({'sample_id':sid,'tokens':len(ids),'bytes':output.stat().st_size}),flush=True)
    out={'model':'Qwen/Qwen2-7B-Instruct','revision':'f2826a00ceef68f0f2b946d945ecc0477ce4450c',
        'layer':14, 'hidden_dim':3584,'selection':'First three MATH entries in the previously frozen 96-case list; chosen before reading vector metrics; no correctness selection.',
        'scope':'Native all-response teacher-forced states from one pass. Includes EOS. Prefix16 means computed from the same pass, not mixed with re-prefill captures. No normalization.',
        'source_cases_sha256':digest(study/'cases.json'),'cases':rows}
    (args.output/'examples.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
