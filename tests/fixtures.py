import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def openact_shards(root, n=20):
    root = Path(root)
    for shard in range(2):
        p = root / f"shard_{shard * n:05d}_{(shard + 1) * n:05d}"
        p.mkdir(parents=True)
        rng = np.random.default_rng(shard + 3)
        labels = np.arange(n) % 2
        hidden = (
            rng.normal(size=(n, 6, 3, 5)).astype("f4") + labels[:, None, None, None] * 2
        )
        prompt = hidden[:, 0].copy()
        z = zarr.open_group(str(p / "tensors.zarr"), mode="w")
        for key, arr in {
            "hidden_states/per_token": hidden.reshape(n * 6, 3, 5),
            "hidden_states/mean": hidden.mean(1),
            "hidden_states/prompt_last": prompt,
            "final_norm/pre/per_token": hidden[:, :, -1].reshape(n * 6, 5) * 3,
            "final_norm/pre/mean": hidden[:, :, -1].mean(1) * 3,
            "final_norm/pre/prompt_last": prompt[:, -1] * 3,
            "tokens/ids": np.arange(n * 6),
            "tokens/sample_ptr": np.arange(n + 1) * 6,
            "tokens/offsets": np.tile(
                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 0]], (n, 1)
            ),
            "metrics/token_entropy": np.linspace(0.1, 1, n * 6),
            "metrics/token_logprob": -np.linspace(0.1, 1, n * 6),
        }.items():
            z.create_dataset(key, data=arr)
        pd.DataFrame(
            {
                "sample_idx": np.arange(n),
                "sample_id": [f"math_{shard * n + i}" for i in range(n)],
                "status": 1,
                "response_text": "a. b.",
                "finish_reason": "eos",
                "category": np.where(labels, "algebra", "geometry"),
                "level": labels + 1,
                "language": "en",
            }
        ).to_parquet(p / "data.parquet")
        (p / "labels").mkdir()
        pd.DataFrame(
            {"sample_idx": np.arange(n), "is_correct": labels.astype(bool)}
        ).to_parquet(p / "labels/correctness.parquet")
        (p / "manifest.json").write_text(
            json.dumps(
                {
                    "model": {
                        "identifier": "fixture",
                        "revision": "pinned",
                        "n_layers": 3,
                        "n_decoder_layers": 2,
                        "hidden_dim": 5,
                    },
                    "capture": {"hidden_states_layers": None},
                }
            )
        )
        (p / "_SUCCESS").write_text("{}")
        (p / "_COPY_VERIFIED.json").write_text("{}")
    return root
