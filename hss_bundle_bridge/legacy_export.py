from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import zarr

from .bundle import FrameSink, compact_sample_frame, ensure_dir, load_frame, now_utc, open_memmap_npy, save_frame, write_bundle_manifest, write_view_manifest
from .protocol import BundleManifest, SelectionSpec, ViewManifest


try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class LegacyRunExporter:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        model: str,
        dataset: str,
        language: str = "en",
        bundle_dir: str | Path,
        dtype: str = "float32",
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.model = str(model)
        self.dataset = str(dataset)
        self.language = str(language)
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.dtype = str(dtype)
        self._zgroup = None
        self._sample_frame: Optional[pd.DataFrame] = None

    def export_bundle(
        self,
        *,
        views: Sequence[str],
        selection: Optional[SelectionSpec] = None,
        overwrite: bool = False,
        sentence_cache_dir: Optional[str | Path] = None,
        tokenizer_name_or_path: Optional[str] = None,
        token_batch_size: int = 64,
        item_batch_size: int = 1024,
    ) -> BundleManifest:
        if self.bundle_dir.exists() and overwrite:
            for child in self.bundle_dir.iterdir():
                if child.is_dir():
                    for sub in sorted(child.rglob("*"), reverse=True):
                        if sub.is_file() or sub.is_symlink():
                            sub.unlink()
                        elif sub.is_dir():
                            sub.rmdir()
                    child.rmdir()
                else:
                    child.unlink()
        ensure_dir(self.bundle_dir)
        ensure_dir(self.bundle_dir / "views")
        sample_frame = self._selected_sample_frame(selection)
        sample_table_path = save_frame(sample_frame, self.bundle_dir / "samples.parquet")
        view_map: Dict[str, str] = {}
        for view in views:
            name = str(view)
            if name == "sample_mean":
                manifest = self._export_sample_array_view(
                    view_name="sample_mean",
                    array_name="mean_answer_hs",
                    sample_frame=sample_frame,
                    item_batch_size=item_batch_size,
                )
            elif name == "sample_prompt_last":
                manifest = self._export_sample_array_view(
                    view_name="sample_prompt_last",
                    array_name="prompt_last_hs",
                    sample_frame=sample_frame,
                    item_batch_size=item_batch_size,
                )
            elif name == "sentence_mean":
                manifest = self._export_sentence_view(
                    sample_frame=sample_frame,
                    sentence_cache_dir=sentence_cache_dir,
                    item_batch_size=item_batch_size,
                )
            elif name == "token":
                manifest = self._export_token_view(
                    sample_frame=sample_frame,
                    tokenizer_name_or_path=tokenizer_name_or_path,
                    token_batch_size=token_batch_size,
                )
            else:
                raise ValueError(f"Unsupported view: {name}")
            relative_path = f"views/{manifest.name}/view.json"
            write_view_manifest(self.bundle_dir, relative_path, manifest)
            view_map[manifest.name] = relative_path
        bundle_manifest = BundleManifest(
            schema_version=1,
            bundle_type="hss_bundle",
            source={
                "source_type": "legacy_lmd_run",
                "run_dir": str(self.run_dir),
                "model": self.model,
                "dataset": self.dataset,
                "language": self.language,
            },
            sample_table_path=str(sample_table_path.relative_to(self.bundle_dir)),
            views=view_map,
            created_at=now_utc(),
            selection=None if selection is None else selection.to_dict(),
            extra={
                "dtype": self.dtype,
                "n_samples": int(sample_frame["sample_id"].nunique()),
            },
        )
        write_bundle_manifest(self.bundle_dir, bundle_manifest)
        return bundle_manifest

    def _selected_sample_frame(self, selection: Optional[SelectionSpec]) -> pd.DataFrame:
        frame = self.sample_frame().copy()
        if selection is None:
            return frame.reset_index(drop=True)
        if selection.sample_ids is not None:
            wanted = {int(x) for x in selection.sample_ids}
            frame = frame[frame["sample_id"].isin(wanted)]
        if selection.query:
            frame = frame.query(selection.query, engine="python")
        if selection.frac is not None:
            frac = float(selection.frac)
            if frac <= 0:
                frame = frame.iloc[0:0].copy()
            elif frac < 1.0 and len(frame) > 0:
                frame = frame.sample(frac=frac, random_state=int(selection.random_state)).sort_values("zarr_row")
        if selection.head is not None:
            frame = frame.head(int(selection.head))
        if selection.tail is not None:
            frame = frame.tail(int(selection.tail))
        return frame.reset_index(drop=True)

    def sample_frame(self) -> pd.DataFrame:
        if self._sample_frame is not None:
            return self._sample_frame.copy()
        parquet_dir = self._parquet_dir()
        metadata_path = parquet_dir / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(str(metadata_path))
        metadata = pd.read_parquet(metadata_path)
        outputs_path = parquet_dir / "outputs.parquet"
        outputs = pd.read_parquet(outputs_path) if outputs_path.exists() else pd.DataFrame()
        if len(outputs) > 0 and "sample_id" not in outputs.columns and "sample_id" in metadata.columns:
            outputs = outputs.copy()
            outputs["sample_id"] = metadata["sample_id"]
        if len(outputs) > 0 and "sample_id" in outputs.columns:
            outputs = outputs.drop_duplicates(subset=["sample_id"], keep="last")
            frame = metadata.merge(outputs, on="sample_id", how="left", suffixes=("", "_out"))
        else:
            frame = metadata.copy()
        mapping = self._sample_mapping()
        frame = frame[frame["sample_id"].isin(mapping.keys())].copy()
        frame["zarr_row"] = frame["sample_id"].map(mapping).astype(np.int64)
        if "is_correct" in frame.columns:
            frame["is_correct"] = frame["is_correct"].astype(np.int32)
        elif "label" in frame.columns:
            frame["is_correct"] = frame["label"].astype(np.int32)
        else:
            frame["is_correct"] = 0
        frame = compact_sample_frame(frame)
        frame = frame.sort_values("zarr_row").reset_index(drop=True)
        self._sample_frame = frame
        return frame.copy()

    def _export_sample_array_view(
        self,
        *,
        view_name: str,
        array_name: str,
        sample_frame: pd.DataFrame,
        item_batch_size: int,
    ) -> ViewManifest:
        array = self.zgroup()[array_name]
        rows = sample_frame["zarr_row"].to_numpy(dtype=np.int64)
        states_path = self.bundle_dir / "views" / view_name / "states.npy"
        index_path = self.bundle_dir / "views" / view_name / "index.parquet"
        ensure_dir(states_path.parent)
        n_items = int(rows.shape[0])
        n_layers = int(array.shape[1])
        hidden_dim = int(array.shape[2])
        mmap = open_memmap_npy(states_path, shape=(n_items, n_layers, hidden_dim), dtype=self.dtype)
        for start in range(0, n_items, max(1, int(item_batch_size))):
            end = min(n_items, start + max(1, int(item_batch_size)))
            batch_rows = rows[start:end]
            mmap[start:end] = self._take_rows(array, batch_rows).astype(self.dtype, copy=False)
        del mmap
        index_frame = pd.DataFrame(
            {
                "item_id": np.arange(n_items, dtype=np.int64),
                "sample_id": sample_frame["sample_id"].to_numpy(dtype=np.int64),
            }
        )
        final_index = save_frame(index_frame, index_path)
        return ViewManifest(
            name=view_name,
            unit="sample",
            states_path=str(states_path.relative_to(self.bundle_dir)),
            index_path=str(final_index.relative_to(self.bundle_dir)),
            n_items=n_items,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dtype=self.dtype,
            layer_ids=list(range(n_layers)),
            extra={"source_array": array_name},
        )

    def _export_sentence_view(
        self,
        *,
        sample_frame: pd.DataFrame,
        sentence_cache_dir: Optional[str | Path],
        item_batch_size: int,
    ) -> ViewManifest:
        cache_dir = self._resolve_sentence_cache_dir(sentence_cache_dir)
        sentence_meta = pd.read_parquet(cache_dir / "sentence_metadata.parquet")
        sentence_meta = sentence_meta[sentence_meta["sample_id"].isin(sample_frame["sample_id"].tolist())].copy()
        sentence_meta = sentence_meta.sort_values("sentence_row").reset_index(drop=True)
        if len(sentence_meta) == 0:
            raise ValueError("sentence cache selection is empty")
        zgroup = zarr.open_group(str(cache_dir / "zarr"), mode="r")
        array = zgroup["sentence_mean_hs"]
        rows = sentence_meta["sentence_row"].to_numpy(dtype=np.int64)
        states_path = self.bundle_dir / "views" / "sentence_mean" / "states.npy"
        index_path = self.bundle_dir / "views" / "sentence_mean" / "index.parquet"
        ensure_dir(states_path.parent)
        n_items = int(rows.shape[0])
        n_layers = int(array.shape[1])
        hidden_dim = int(array.shape[2])
        mmap = open_memmap_npy(states_path, shape=(n_items, n_layers, hidden_dim), dtype=self.dtype)
        for start in range(0, n_items, max(1, int(item_batch_size))):
            end = min(n_items, start + max(1, int(item_batch_size)))
            batch_rows = rows[start:end]
            mmap[start:end] = self._take_rows(array, batch_rows).astype(self.dtype, copy=False)
        del mmap
        keep_cols = [
            c
            for c in [
                "sample_id",
                "sentence_row",
                "sentence_index",
                "sentence_text",
                "sentence_char_start",
                "sentence_char_end",
                "token_start",
                "token_end",
                "n_sentence_tokens",
                "alignment_status",
            ]
            if c in sentence_meta.columns
        ]
        index_frame = sentence_meta[keep_cols].copy()
        index_frame.insert(0, "item_id", np.arange(len(index_frame), dtype=np.int64))
        final_index = save_frame(index_frame, index_path)
        return ViewManifest(
            name="sentence_mean",
            unit="sentence",
            states_path=str(states_path.relative_to(self.bundle_dir)),
            index_path=str(final_index.relative_to(self.bundle_dir)),
            n_items=n_items,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dtype=self.dtype,
            layer_ids=list(range(n_layers)),
            extra={"sentence_cache_dir": str(cache_dir)},
        )

    def _export_token_view(
        self,
        *,
        sample_frame: pd.DataFrame,
        tokenizer_name_or_path: Optional[str],
        token_batch_size: int,
    ) -> ViewManifest:
        if "answer_tok_values" not in self.zgroup():
            raise KeyError("answer_tok_values is not available in the legacy run")
        values = self.zgroup()["answer_tok_values"]
        ptr = np.asarray(self.zgroup()["answer_tok_ptr"][:], dtype=np.int64)
        token_ids_ds = self.zgroup()["answer_tok_ids"] if "answer_tok_ids" in self.zgroup() else None
        tokenizer = None
        if tokenizer_name_or_path is not None:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for token decoding")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=True)
        selected = sample_frame.sort_values("zarr_row").reset_index(drop=True)
        rows = selected["zarr_row"].to_numpy(dtype=np.int64)
        counts = ptr[rows + 1] - ptr[rows]
        total_tokens = int(counts.sum())
        n_layers = int(values.shape[1])
        hidden_dim = int(values.shape[2])
        states_path = self.bundle_dir / "views" / "token" / "states.npy"
        index_path = self.bundle_dir / "views" / "token" / "index.parquet"
        ensure_dir(states_path.parent)
        mmap = open_memmap_npy(states_path, shape=(total_tokens, n_layers, hidden_dim), dtype=self.dtype)
        sink = FrameSink(index_path)
        cursor = 0
        batch_size = max(1, int(token_batch_size))
        sample_lookup = selected.set_index("zarr_row")
        for batch_start in range(0, len(rows), batch_size):
            batch_end = min(len(rows), batch_start + batch_size)
            batch_rows = rows[batch_start:batch_end]
            ptr_start = int(ptr[int(batch_rows[0])])
            ptr_end = int(ptr[int(batch_rows[-1]) + 1])
            token_block = np.asarray(values[ptr_start:ptr_end], dtype=self.dtype)
            token_ids_block = None
            if token_ids_ds is not None:
                token_ids_block = np.asarray(token_ids_ds[ptr_start:ptr_end], dtype=np.int64)
            frames: List[pd.DataFrame] = []
            for zarr_row in batch_rows:
                sample_row = sample_lookup.loc[int(zarr_row)]
                a = int(ptr[int(zarr_row)] - ptr_start)
                b = int(ptr[int(zarr_row) + 1] - ptr_start)
                seq = token_block[a:b]
                n_tok = int(seq.shape[0])
                if n_tok == 0:
                    continue
                mmap[cursor:cursor + n_tok] = seq
                token_texts, char_starts, char_ends = self._decode_token_texts(
                    sample_row=sample_row,
                    n_tokens=n_tok,
                    tokenizer=tokenizer,
                    token_ids=None if token_ids_block is None else token_ids_block[a:b],
                )
                frame = pd.DataFrame(
                    {
                        "item_id": np.arange(cursor, cursor + n_tok, dtype=np.int64),
                        "sample_id": np.full(n_tok, int(sample_row["sample_id"]), dtype=np.int64),
                        "token_position": np.arange(n_tok, dtype=np.int32),
                        "token_text": token_texts if token_texts is not None else [""] * n_tok,
                        "char_start": np.asarray(char_starts, dtype=np.float64) if char_starts is not None else np.full(n_tok, np.nan, dtype=np.float64),
                        "char_end": np.asarray(char_ends, dtype=np.float64) if char_ends is not None else np.full(n_tok, np.nan, dtype=np.float64),
                    }
                )
                frames.append(frame)
                cursor += n_tok
            if frames:
                sink.write(pd.concat(frames, ignore_index=True))
        del mmap
        final_index = sink.close()
        return ViewManifest(
            name="token",
            unit="token",
            states_path=str(states_path.relative_to(self.bundle_dir)),
            index_path=str(final_index.relative_to(self.bundle_dir)),
            n_items=total_tokens,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dtype=self.dtype,
            layer_ids=list(range(n_layers)),
            extra={"has_token_text": tokenizer is not None or token_ids_ds is not None},
        )

    def zgroup(self) -> zarr.hierarchy.Group:
        if self._zgroup is None:
            self._zgroup = zarr.open_group(str(self._zarr_path()), mode="r")
        return self._zgroup

    def _zarr_path(self) -> Path:
        path = self.run_dir / "zarr" / f"{self.model}__{self.dataset}__{self.language}"
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

    def _parquet_dir(self) -> Path:
        with_lang = self.run_dir / "parquet" / self.model / self.dataset / self.language
        without_lang = self.run_dir / "parquet" / self.model / self.dataset
        if with_lang.exists():
            return with_lang
        if without_lang.exists():
            return without_lang
        return with_lang

    def _resolve_sentence_cache_dir(self, sentence_cache_dir: Optional[str | Path]) -> Path:
        if sentence_cache_dir is not None:
            path = Path(sentence_cache_dir).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(str(path))
            return path
        candidates = [
            self.run_dir / "sentence_experiment" / "sentence_cache",
            self.run_dir / "sentence_cache",
        ]
        for path in candidates:
            if path.exists():
                return path.resolve()
        raise FileNotFoundError("Could not find a sentence cache directory")

    def _sample_mapping(self) -> Dict[int, int]:
        sample_ids = np.asarray(self.zgroup()["sample_id"][:], dtype=np.int64)
        if "sample_valid" in self.zgroup():
            valid = np.asarray(self.zgroup()["sample_valid"][:], dtype=bool)
        else:
            valid = np.ones(sample_ids.shape[0], dtype=bool)
        return {int(sample_id): int(row) for row, sample_id in enumerate(sample_ids) if valid[row] and int(sample_id) >= 0}

    def _take_rows(self, array: Any, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if hasattr(array, "oindex"):
            return np.asarray(array.oindex[rows], dtype=self.dtype)
        return np.asarray(array[rows], dtype=self.dtype)

    def _decode_token_texts(
        self,
        *,
        sample_row: pd.Series,
        n_tokens: int,
        tokenizer: Any,
        token_ids: Optional[np.ndarray],
    ) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[List[int]]]:
        if token_ids is not None and tokenizer is not None:
            try:
                tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
                return [str(x) for x in tokens], None, None
            except Exception:
                pass
        text = str(sample_row.get("generated_text", "") or sample_row.get("text", "") or "")
        if tokenizer is None or not text:
            return None, None, None
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            offsets = encoded.get("offset_mapping", [])
            if len(offsets) != n_tokens:
                return None, None, None
            pieces: List[str] = []
            starts: List[int] = []
            ends: List[int] = []
            for a, b in offsets:
                a = int(a)
                b = int(b)
                pieces.append(text[a:b])
                starts.append(a)
                ends.append(b)
            return pieces, starts, ends
        except Exception:
            return None, None, None


def export_legacy_run_bundle(
    *,
    run_dir: str | Path,
    model: str,
    dataset: str,
    language: str,
    bundle_dir: str | Path,
    views: Sequence[str],
    selection: Optional[SelectionSpec] = None,
    overwrite: bool = False,
    sentence_cache_dir: Optional[str | Path] = None,
    tokenizer_name_or_path: Optional[str] = None,
    dtype: str = "float32",
) -> BundleManifest:
    exporter = LegacyRunExporter(
        run_dir=run_dir,
        model=model,
        dataset=dataset,
        language=language,
        bundle_dir=bundle_dir,
        dtype=dtype,
    )
    return exporter.export_bundle(
        views=views,
        selection=selection,
        overwrite=overwrite,
        sentence_cache_dir=sentence_cache_dir,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
