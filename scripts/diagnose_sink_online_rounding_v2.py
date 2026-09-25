"""Replay the second stopped operation; numerical checks only, no outcome search."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from extract_revision_prefixes import load_model
from run_hss_followup import generate
from run_sink_next_generation_v2 import OnlineTransform
from run_sink_energy_matched_v3 import MatchedTransform
from revision_common import write_json, write_npz, sha


def main():
    root = Path('/lambda/nfs/dami/hss/sink-next-20260925/exp4-v2')
    out = root/'rounding_diagnostic'; out.mkdir(exist_ok=True)
    cfg = json.loads((root/'plan.json').read_text())['config']
    torch.set_num_threads(4)
    v = dict(np.load('/lambda/nfs/dami/hss/sink-energy-matched-20260925-v3/directions.npz'))
    path = out/'failed_activation.npz'
    if not path.exists():
        model, tok = load_model(cfg)
        class Capture(OnlineTransform):
            def __call__(self, h):
                try:
                    return super().__call__(h)
                except AssertionError:
                    write_npz(path, h=h.float().cpu().numpy())
                    raise
        row = next(r for r in json.loads((root/'cohort.json').read_text()) if r['sample_id']=='math_2147')
        gen = copy.deepcopy(model.generation_config)
        gen.do_sample=False; gen.num_beams=1; gen.repetition_penalty=1.
        if gen.pad_token_id is None:
            gen.pad_token_id=tok.pad_token_id or tok.eos_token_id
        op = Capture(v['axis'], float(v['tau']), v['orthogonal_controls'][5], cfg, model.device)
        try:
            generate(model, tok, row['prompt_ids'], [], gen, 2048, op)
        except AssertionError:
            assert path.exists()
        else:
            raise RuntimeError('Expected stopped operation did not reproduce')
        del model
    h = torch.as_tensor(np.load(path)['h'], device='cuda').to(torch.bfloat16)
    results=[]
    for steps in [32, 128, 512, 2048]:
        test=dict(cfg, internal_absolute_norm_target=0., rounding_repair_steps=steps)
        op=MatchedTransform(v['axis'], float(v['tau']), v['orthogonal_controls'][5], test, h.device)
        passed=True
        try:
            op(h)
        except AssertionError:
            passed=False
        results.append(dict(steps=steps, passed=passed, metrics=op.metrics))
    write_json(out/'summary.json', dict(failed_activation_sha256=sha(path), results=results,
        scope='Only the numerical operation; no logits or generation outcome comparison.'))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
