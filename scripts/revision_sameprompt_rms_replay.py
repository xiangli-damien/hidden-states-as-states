"""Exact row-local RMS replay, preserving full-sequence CUDA reduction geometry."""
import numpy as np


def exact_replay(raw, saved, positions, gamma, eps):
    import torch
    raw, saved, gamma = np.asarray(raw), np.asarray(saved), np.asarray(gamma)
    assert raw.ndim == 2 and saved.shape == raw.shape and gamma.shape == (raw.shape[1],)
    assert len(positions) == len(raw) and positions == list(range(positions[0], positions[-1]+1))
    assert np.isfinite(raw).all() and np.isfinite(saved).all() and np.isfinite(gamma).all()
    with torch.inference_mode():
        # The collection norm consumed [1, prompt_length + prefix_length, D].
        # Unrecorded rows may be zero: RMSNorm has no cross-token dependence.
        h = torch.zeros((1, positions[-1]+1, raw.shape[1]), dtype=torch.bfloat16, device='cuda')
        values = torch.from_numpy(raw.copy()).to(device='cuda', dtype=torch.bfloat16)
        np.testing.assert_array_equal(values.float().cpu().numpy(), raw)
        h[0, positions] = values
        weight = torch.from_numpy(gamma.copy()).to(device='cuda', dtype=torch.bfloat16)
        np.testing.assert_array_equal(weight.float().cpu().numpy(), gamma)
        x = h.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        normalized = x*torch.rsqrt(variance+eps)
        result = weight*normalized.to(torch.bfloat16)
        actual = result[0, positions].float().cpu().numpy()
    # This is stricter than the failed CPU tolerance check. Wrong positions,
    # weights, dtype, or data cannot pass merely through a relaxed threshold.
    np.testing.assert_array_equal(actual, saved)
    return {'checked_elements': int(saved.size), 'maximum_error': 0.}


def self_check():
    """Real CUDA HF RMSNorm with other rows present, plus a corruption rejection."""
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    torch.manual_seed(42)
    norm = Qwen2RMSNorm(3584, eps=1e-6).to(device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        norm.weight.copy_(torch.randn_like(norm.weight)*4)
        whole = (torch.randn((1, 173, 3584), device='cuda')*30).to(torch.bfloat16)
        saved = norm(whole).float().cpu().numpy()
    gamma = norm.weight.detach().float().cpu().numpy(); x = whole.float().cpu().numpy()
    checked = []
    for positions in [[172], list(range(109, 173)), list(range(157, 173))]:
        result = exact_replay(x[0, positions], saved[0, positions], positions, gamma, 1e-6)
        checked.append(result['checked_elements'])
    wrong = saved[0, [172]].copy(); wrong[0, 10] += .125
    try:
        exact_replay(x[0, [172]], wrong, [172], gamma, 1e-6)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted post-norm value accepted')
    return {'passed': True, 'actual_hf_cuda_rmsnorm_exact': True,
            'other_rows_have_no_effect': True, 'corruption_rejected': True,
            'checked_elements': sum(checked)}


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from revision_common import write_json, sha
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = self_check(); result['source_sha256'] = sha(Path(__file__))
    write_json(args.output, result); print(json.dumps(result))
