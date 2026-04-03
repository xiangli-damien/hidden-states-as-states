from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .adapter import BundleAdapter
from .legacy_export import export_legacy_run_bundle
from .protocol import SelectionSpec


def _selection_from_args(args: argparse.Namespace) -> Optional[SelectionSpec]:
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x) for x in args.sample_ids.split(",") if str(x).strip()]
    if not any([sample_ids, args.query, args.head is not None, args.tail is not None, args.frac is not None]):
        return None
    return SelectionSpec(
        sample_ids=sample_ids,
        query=args.query,
        head=args.head,
        tail=args.tail,
        frac=args.frac,
        random_state=args.random_state,
    )


def _cmd_export(args: argparse.Namespace) -> None:
    selection = _selection_from_args(args)
    manifest = export_legacy_run_bundle(
        run_dir=args.run_dir,
        model=args.model,
        dataset=args.dataset,
        language=args.language,
        bundle_dir=args.bundle_dir,
        views=args.views,
        selection=selection,
        overwrite=args.overwrite,
        sentence_cache_dir=args.sentence_cache_dir,
        tokenizer_name_or_path=args.tokenizer,
        dtype=args.dtype,
    )
    print(json.dumps(manifest.to_dict(), indent=2))


def _cmd_inspect(args: argparse.Namespace) -> None:
    adapter = BundleAdapter(args.bundle_dir)
    payload = {
        "bundle": adapter.bundle_manifest.to_dict(),
        "views": {},
    }
    for name in adapter.list_views():
        manifest = adapter.view_manifest(name)
        payload["views"][name] = manifest.to_dict()
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--run-dir", required=True)
    export_parser.add_argument("--model", required=True)
    export_parser.add_argument("--dataset", required=True)
    export_parser.add_argument("--language", default="en")
    export_parser.add_argument("--bundle-dir", required=True)
    export_parser.add_argument("--views", nargs="+", required=True)
    export_parser.add_argument("--sentence-cache-dir", default=None)
    export_parser.add_argument("--tokenizer", default=None)
    export_parser.add_argument("--dtype", default="float32")
    export_parser.add_argument("--sample-ids", default=None)
    export_parser.add_argument("--query", default=None)
    export_parser.add_argument("--head", type=int, default=None)
    export_parser.add_argument("--tail", type=int, default=None)
    export_parser.add_argument("--frac", type=float, default=None)
    export_parser.add_argument("--random-state", type=int, default=42)
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.set_defaults(func=_cmd_export)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--bundle-dir", required=True)
    inspect_parser.set_defaults(func=_cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
