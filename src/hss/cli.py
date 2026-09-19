"""The supported CLI for OpenAct-backed HSS reproduction and sweeps."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hss")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "plan", "run", "sweep"):
        p = sub.add_parser(command)
        p.add_argument("config")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=JSON",
            help="Typed dotted override; quote string values",
        )
        p.add_argument(
            "--refresh-data",
            action="store_true",
            help="Explicitly replace the saved sweep data snapshot",
        )
    plot = sub.add_parser("report")
    plot.add_argument("output_root")
    plot.add_argument("--destination", default=None)
    plot.add_argument("--formats", nargs="+", default=["png", "svg"])
    plot.add_argument("--bootstrap", type=int, default=0)
    sub.add_parser("methods")
    data = sub.add_parser("data")
    data.add_argument("catalog")
    results = sub.add_parser("results")
    results.add_argument("roots", nargs="+")
    results.add_argument("--validate", action="store_true")
    results.add_argument(
        "--full", action="store_true", help="Verify all SHA-256 checksums"
    )
    figures = sub.add_parser("figures")
    figures.add_argument("roots", nargs="+")
    figures.add_argument("--suite", required=True)
    figures.add_argument("--destination", required=True)
    figures.add_argument("--only", nargs="+")
    figures.add_argument("--formats", nargs="+", default=["png", "svg"])
    figures.add_argument("--max-trajectories", type=int, default=1000)
    figures.add_argument("--seed", type=int, default=42)
    figures.add_argument(
        "--linkage", choices=["average", "complete", "single"], default="average"
    )
    figures.add_argument("--check", action="store_true")
    figures.add_argument("--strict", action="store_true")
    controls = sub.add_parser("controls")
    controls.add_argument("roots", nargs="+")
    controls.add_argument("--suite", required=True)
    controls.add_argument("--destination", required=True)
    controls.add_argument("--only", nargs="+")
    controls.add_argument("--formats", nargs="+", default=["png", "svg"])
    controls.add_argument("--bootstrap", type=int, default=0)
    paper = sub.add_parser("paper")
    paper.add_argument("--data-root", default="/lambda/nfs/dami/openact/runs")
    paper.add_argument(
        "--catalog",
        help="Named model/dataset catalog; overrides legacy data-root paths",
    )
    paper.add_argument("--directory", required=True)
    paper.add_argument("--cache-root", default="/home/ubuntu/hss-cache")
    paper.add_argument("--output-root", default="/lambda/nfs/dami/hss/results/trials")
    paper.add_argument(
        "--artifact-cache-root", default="/lambda/nfs/dami/hss/cache/fits"
    )
    paper.add_argument("--check-data", action="store_true")
    paper.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly regenerate existing study configs",
    )
    suite = sub.add_parser("suite")
    suite.add_argument("manifest")
    suite.add_argument("--only", nargs="+")
    suite.add_argument("--set", action="append", default=[])
    suite.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            from .cluster.methods import METHODS

            result = METHODS
        elif args.command == "data":
            from .data.catalog import inspect_catalog

            result = inspect_catalog(args.catalog)
        elif args.command == "results":
            from .results import ResultCatalog

            catalog = ResultCatalog(args.roots)
            result = (
                catalog.validate(args.full)
                if args.validate or args.full
                else {"results": [r.metadata() for r in catalog.results]}
            )
        elif args.command == "controls":
            from .viz.controls import render_controls

            result = render_controls(
                args.roots,
                args.suite,
                args.destination,
                args.only,
                args.formats,
                args.bootstrap,
            )
        elif args.command == "figures":
            from .viz.paper import render_paper

            result = render_paper(
                args.roots,
                args.suite,
                args.destination,
                args.only,
                args.formats,
                args.max_trajectories,
                args.seed,
                args.linkage,
                args.check,
                args.strict,
            )
        elif args.command == "suite":
            from .experiments.paper import run_suite

            result = run_suite(args.manifest, args.only, args.set, args.refresh_data)
        elif args.command == "paper":
            from .experiments.paper import generate_suite

            result = generate_suite(
                args.data_root,
                args.directory,
                args.cache_root,
                args.output_root,
                args.check_data,
                args.catalog,
                args.artifact_cache_root,
                args.overwrite,
            )
        elif args.command == "report":
            from .experiments.reporting import report

            result = report(
                args.output_root,
                args.destination,
                formats=args.formats,
                bootstrap=args.bootstrap,
            )
        else:
            from .experiments.config import load

            cfg = load(args.config, args.set)
            if args.command == "prepare":
                from .experiments.openact import prepare

                data = prepare(cfg.data, cfg.execution.cache_root)
                result = {"path": str(data.path), "snapshot": data.info}
            elif args.command == "plan":
                from .experiments.sweep import plan

                result = plan(cfg, refresh_data=args.refresh_data)
            elif args.command == "sweep":
                from .experiments.sweep import run_sweep

                result = run_sweep(cfg, refresh_data=args.refresh_data)
            else:
                from .experiments.runner import run_experiment

                result = run_experiment(cfg)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("status") == "failed" else 0
    except (ValueError, TypeError, OSError, MemoryError, RuntimeError) as exc:
        parser.exit(2, f"hss: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
