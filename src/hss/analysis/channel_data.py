"""Bounded, resumable CPU extraction for the residual-channel study.

Reads verified published shards only. t1 means the state AFTER consuming the
first generated token; prompt_last is the state which predicts that token.
Missing late positions stay NaN, never silently repeat a response's last token.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import zarr

from hss.experiments.artifacts import digest, file_digest, save_json


def load_config(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def source_identity(path):
    required = ["manifest.json", "data.parquet", "labels/correctness.parquet",
                "_SUCCESS", "_COPY_VERIFIED.json"]
    return {name: file_digest(path / name) for name in required}


def canonical_questions(frame):
    """Choose a canonical prompt occurrence without inspecting correctness labels."""
    keys = frame.prompt_text.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    hashed = keys.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    return pd.DataFrame({"sample_id":frame.sample_id.astype(str), "prompt_sha256":hashed,
                         "canonical_question":~hashed.duplicated(keep="first")})


def prepare_question_identity(cfg):
    for ds in cfg["datasets"]:
        root = Path(cfg["cache_root"])/ds["name"]
        summary_marker = root/"summary"/"_SUCCESS.json"
        if not summary_marker.exists():
            continue
        info = json.loads(summary_marker.read_text())
        key = digest({"sources":info["source_keys"],"rule":"normalized_prompt_first_occurrence_v1"})
        marker = root/"question_identity.json"
        if marker.exists() and json.loads(marker.read_text())["key"] == key:
            continue
        frames = [pd.read_parquet(Path(ds["path"])/shard/"data.parquet",columns=["sample_id","prompt_text","ground_truth"]) for shard in info["shards"]]
        frame = pd.concat(frames,ignore_index=True)
        identities = canonical_questions(frame)
        conflicts = pd.DataFrame({"prompt":identities.prompt_sha256,"gold":frame.ground_truth.astype(str)}).groupby("prompt").gold.nunique()
        identities.to_parquet(root/"question_identity.parquet",index=False)
        save_json(marker,{"key":key,"original_n":len(frame),"unique_prompts":int(identities.canonical_question.sum()),
                          "duplicate_extra_rows":int((~identities.canonical_question).sum()),
                          "duplicate_groups_with_conflicting_gold":int((conflicts>1).sum()),
                          "rule":"First occurrence in sorted published shard/row order, independent of labels."})


def read_rows(path, z):
    frame = pd.read_parquet(path / "data.parquet")
    labels = pd.read_parquet(path / "labels/correctness.parquet")
    if frame.sample_idx.duplicated().any() or labels.sample_idx.duplicated().any():
        raise ValueError("Duplicate within-shard indices")
    labels = labels.set_index("sample_idx").reindex(frame.sample_idx)
    if labels.is_correct.isna().any() or not labels.is_correct.isin([0, 1]).all():
        raise ValueError("Missing/non-binary correctness")
    if "meta_sample_id" in labels and not np.array_equal(
        labels.meta_sample_id.astype(str), frame.sample_id.astype(str)
    ):
        raise ValueError("Correctness identity mismatch")
    ptr = np.asarray(z["tokens/sample_ptr"][:], dtype=np.int64)
    if len(ptr) != len(frame) + 1 or ptr[0] != 0 or np.any(np.diff(ptr) <= 0):
        raise ValueError("Invalid token pointers")
    if not (frame.status == 1).all():
        raise ValueError("Incomplete sample")
    if "n_response_tokens" in frame and not np.array_equal(
        frame.n_response_tokens, np.diff(ptr)
    ):
        raise ValueError("Token count mismatch")
    rows = pd.DataFrame({
        "sample_id": frame.sample_id.astype(str), "y": labels.is_correct.to_numpy(int),
        "category": frame.get("category", frame.get("subject", "unknown")),
        "level": frame.get("level", 0),
        "n_prompt_tokens": frame.get("n_prompt_tokens", 0),
        "n_tokens": np.diff(ptr), "truncated": frame.finish_reason.eq("length"),
        "parse_failed": labels.get("meta_parse_failed", pd.Series(False, index=labels.index)).to_numpy(bool),
        "first_token": z["tokens/ids"].oindex[ptr[:-1]],
        "opening": frame.response_text.str[:100],
        "shard": path.name,
    })
    return rows, ptr


def token_samples(array, ptr, positions):
    """One orthogonal read per shard, so each selected chunk decompresses once."""
    lengths = np.diff(ptr)
    offsets = np.asarray(positions, dtype=np.int64) - 1
    indices = ptr[:-1, None] + offsets[None, :]
    valid = offsets[None, :] < lengths[:, None]
    indices = np.column_stack([indices, ptr[1:] - 1])
    valid = np.column_stack([valid, np.ones(len(lengths), dtype=bool)])
    unique, inverse = np.unique(indices[valid], return_inverse=True)
    out = np.full((*indices.shape, array.shape[-1]), np.nan, dtype=np.float32)
    out[valid] = array.oindex[unique, :][inverse]
    return out


def extract_shard(job):
    dataset, path, cache, phase, positions = job
    start = time.monotonic()
    identity = source_identity(path)
    dest = cache / dataset["name"] / phase / path.name
    key = digest({"schema": 1, "source": identity, "phase": phase, "positions": positions})
    marker = dest / "_SUCCESS.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["key"] != key:
            raise ValueError(f"Stale cache: {dest}")
        return old
    manifest = json.loads((path / "manifest.json").read_text())
    model = manifest["model"]
    if model["identifier"] != dataset["model"]:
        raise ValueError("Unexpected model")
    capture = manifest.get("capture", {}).get("hidden_states_layers")
    if capture is not None and capture != list(range(model["n_layers"])):
        raise ValueError("This study requires full layer coverage")
    z = zarr.open_consolidated(str(path / "tensors.zarr"), mode="r")
    rows, ptr = read_rows(path, z)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    rows.to_parquet(tmp / "rows.parquet", index=False)
    if phase == "summary":
        for name in ("prompt_last", "mean"):
            np.save(tmp / f"{name}.npy", z[f"hidden_states/{name}"][:], allow_pickle=False)
            np.save(tmp / f"pre_{name}.npy", z[f"final_norm/pre/{name}"][:], allow_pickle=False)
    elif phase == "tokens":
        for norm in ("pre", "post"):
            arr = z[f"final_norm/{norm}/per_token"]
            if arr.shape[0] != ptr[-1]:
                raise ValueError("Hidden-state token alignment mismatch")
            values = token_samples(arr, ptr, positions)
            np.save(tmp / f"{norm}.npy", values, allow_pickle=False)
    else:
        raise ValueError(phase)
    result = {"key": key, "source": identity, "source_path": str(path), "n": len(rows),
              "model": model, "positions": positions + ["last"],
              "alignment": manifest.get("custom", {}), "elapsed_seconds": time.monotonic()-start}
    save_json(tmp / "_SUCCESS.json", result)
    tmp.rename(dest)
    print(json.dumps({"dataset": dataset["name"], "phase": phase, "shard": path.name,
                      "seconds": round(result["elapsed_seconds"], 2)}), flush=True)
    return result


def extract(cfg, phase):
    cache = Path(cfg["cache_root"])
    for ds in cfg["datasets"]:
        shards = sorted(p.parent for p in Path(ds["path"]).glob("shard_*/_COPY_VERIFIED.json")
                        if (p.parent / "_SUCCESS").is_file())
        jobs = [(ds, p, cache, phase, cfg["token_positions"]) for p in shards]
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            results = list(pool.map(extract_shard, jobs))
        if sum(x["n"] for x in results) != ds["expected_samples"]:
            raise ValueError(f"Incomplete dataset: {ds['name']}")
        models = {digest(x["model"]) for x in results}
        if len(models) != 1:
            raise ValueError("Mixed model revisions")
        root = cache / ds["name"] / phase
        rows = pd.concat([pd.read_parquet(root / p.name / "rows.parquet") for p in shards], ignore_index=True)
        if rows.sample_id.duplicated().any():
            raise ValueError("Duplicate samples across shards")
        rows.to_parquet(root / "rows.parquet", index=False)
        save_json(root / "_SUCCESS.json", {"dataset": ds, "shards": [p.name for p in shards],
                    "model": results[0]["model"], "positions": cfg["token_positions"]+["last"],
                    "source_keys": [x["key"] for x in results]})


class ChannelData:
    def __init__(self, cfg, name, phase="summary"):
        self.root = Path(cfg["cache_root"]) / name / phase
        self.info = json.loads((self.root / "_SUCCESS.json").read_text())
        self.rows = pd.read_parquet(self.root / "rows.parquet")
        self.raw_n = len(self.rows)
        self._indices = np.arange(self.raw_n)
        identity_path = self.root.parent/"question_identity.parquet"
        if identity_path.exists():
            identity = pd.read_parquet(identity_path).set_index("sample_id").loc[self.rows.sample_id]
            self._indices = np.flatnonzero(identity.canonical_question.to_numpy(bool))
            self.rows = self.rows.iloc[self._indices].reset_index(drop=True)
            self.rows["prompt_sha256"] = identity.prompt_sha256.iloc[self._indices].to_numpy()
            self.info["question_identity"] = json.loads((self.root.parent/"question_identity.json").read_text())

    def array(self, name, index=None):
        arrays = []
        for shard in self.info["shards"]:
            a = np.load(self.root / shard / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            arrays.append(np.asarray(a if index is None else a[:, index, :]))
        return np.concatenate(arrays)[self._indices]
