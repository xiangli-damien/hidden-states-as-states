"""Cache branching for independent versus cumulative teacher-forced patches."""
import copy
import numpy as np
from promptlast_replacement_common import apply


def names(methods):
    return ['identity'] + [f'{mode}_{m}' for mode in ['independent', 'cumulative'] for m in methods]


def selected_cases(cases, per_dataset):
    return [r for ds in ['math', 'gsm8k'] for r in [c for c in cases if c['dataset'] == ds][:per_dataset]]


def one_step(model, token, cache, layer, transform=None):
    import torch
    captured = {}
    def hook(module, args, out):
        h = out[0] if isinstance(out, tuple) else out
        assert h.shape[:2] == (1, 1) and not captured
        captured['x'] = h[0].float().cpu().numpy().copy()
        z = h[0] if transform is None else transform(h[0])
        assert z.shape == h[0].shape
        captured['actual'] = z.float().cpu().numpy().copy()
        patched = h.clone(); patched[0] = z
        return (patched,) + out[1:] if isinstance(out, tuple) else patched
    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([[token]], device=model.device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            logp = out.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    finally:
        handle.remove()
    assert captured
    return logp, out.past_key_values, captured


def trajectory(model, prompt, response, decoder, layer, methods):
    """Yield all predictions: p=0 predicts response[0] from prompt[-1].

    Each independent branch starts from a deep copy of the unmodified clean
    cache BEFORE this token. Cumulative branches own their persistent caches.
    All branches consume exactly the same observed tokens, never sampled ones.
    """
    import torch
    assert len(prompt) >= 2 and len(response) > 0
    with torch.inference_mode():
        clean_cache = model.model(input_ids=torch.tensor([prompt[:-1]], device=model.device), use_cache=True).past_key_values
    cumulative = {m: copy.deepcopy(clean_cache) for m in methods}
    order = names(methods)
    for position, target in enumerate(response):
        token = prompt[-1] if position == 0 else response[position - 1]
        outputs = {}
        # The clean cache must not be advanced before the independent branches.
        for method in methods:
            def transform(h, method=method):
                value, _ = apply(h.float().cpu().numpy(), decoder, method)
                return torch.as_tensor(value, device=h.device, dtype=h.dtype)
            before = clean_cache.get_seq_length()
            lp, branch, cap = one_step(model, token, copy.deepcopy(clean_cache), layer, transform)
            assert clean_cache.get_seq_length() == before and branch.get_seq_length() == before + 1
            outputs['independent_' + method] = (lp, cap)
            del branch
            lp, cumulative[method], cap = one_step(model, token, cumulative[method], layer, transform)
            outputs['cumulative_' + method] = (lp, cap)
        lp, clean_cache, cap = one_step(model, token, clean_cache, layer)
        outputs['identity'] = (lp, cap)
        for name in order:
            # Patching a block output cannot affect the lower layers on fixed tokens.
            np.testing.assert_array_equal(outputs[name][1]['x'], cap['x'])
        for method in methods:
            np.testing.assert_array_equal(outputs['independent_' + method][1]['actual'], outputs['cumulative_' + method][1]['actual'])
            if position == 0:
                np.testing.assert_array_equal(outputs['independent_' + method][0], outputs['cumulative_' + method][0])
        yield {'position': position, 'target_id': target, 'input_id': token,
               'logp': np.stack([outputs[n][0] for n in order]), 'x': cap['x'][0],
               'actual': np.stack([outputs[n][1]['actual'][0] for n in order])}


def metrics(logp, x, actual, target, decoder):
    p = np.exp(logp[0].astype(np.float64))
    kl = ((logp[0].astype(np.float64)[None] - logp.astype(np.float64)) * p).sum(-1)
    _, meta = apply(x[None], decoder, 'identity')
    error = np.square(actual.astype(float) - x.astype(float)).sum(-1)
    energy = np.square(x.astype(float) - decoder['train_mean']).sum()
    return {'kl': kl.tolist(), 'reference_nll': (-logp[:, target].astype(float)).tolist(),
            'argmax': logp.argmax(-1).tolist(), 'region': meta['region'], 'distance': meta['distance'],
            'x_norm': float(np.linalg.norm(x.astype(float))), 'nmse': (error / energy).tolist(),
            'entropy': float(-(p * logp[0]).sum())}
