"""Real random-small-Qwen capture/sampling on CPU, not a scientific outcome."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from collect_revision_sameprompt import capture, sample
from revision_common import sha, write_json


def run():
    torch.set_num_threads(2); torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        pad_token_id=0, eos_token_id=30)).eval()
    prompt = [1, 2, 3, 4]; future = [5, 6, 7, 8, 9, 10]
    current = capture(model, prompt+future[:3], [4, 5, 6], [1, 3], 2)
    extended = capture(model, prompt+future, [4, 5, 6], [1, 3], 2)
    for key in ['raw', 'post', 'logits', 'logprobs']:
        np.testing.assert_allclose(current[key], extended[key], rtol=2e-5, atol=2e-6)
    with torch.no_grad():
        direct = model(torch.tensor([prompt+future[:3]]), use_cache=False, logits_to_keep=1).logits[0, -1].numpy()
    np.testing.assert_allclose(current['logits'], direct, rtol=2e-5, atol=2e-6)
    x = current['raw'][-1].astype(np.float64)
    expected = x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+model.config.rms_norm_eps)
    expected *= model.model.norm.weight.detach().numpy()
    np.testing.assert_allclose(expected, current['post'], rtol=2e-5, atol=2e-6)
    initial = capture(model, prompt, [3], [1, 3], 2)
    cfg = model.generation_config
    cfg.do_sample = True; cfg.temperature = .7; cfg.top_p = .95; cfg.top_k = 0
    cfg.repetition_penalty = 1.; cfg.max_new_tokens = 12
    first = sample(model, prompt, 137, cfg)
    sample(model, prompt, 271, cfg)
    second = sample(model, prompt, 137, cfg)
    assert first == second
    after = capture(model, prompt, [3], [1, 3], 2)
    for key in initial:
        np.testing.assert_array_equal(initial[key], after[key])
    return {'passed': True, 'random_small_model_only': True, 'device': 'cpu',
        'prefix_future_invariance': True, 'head_matches_causal_model': True,
        'rms_relation': True, 'per_trajectory_seed_order_independence': True,
        'same_prompt_state_exact': True,
        'source_sha256': sha(Path(__file__).with_name('collect_revision_sameprompt.py'))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = run()
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result))
