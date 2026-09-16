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
    paper = sub.add_parser("paper")
    paper.add_argument("--data-root", required=True)
    paper.add_argument("--directory", required=True)
    paper.add_argument("--cache-root", default="/home/ubuntu/hss-cache")
    paper.add_argument("--output-root", default="/lambda/nfs/dami/hss-results")
    paper.add_argument("--check-data", action="store_true")
    suite = sub.add_parser("suite")
    suite.add_argument("manifest")
    suite.add_argument("--only", nargs="+")
    suite.add_argument("--set", action="append", default=[])
    suite.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "suite":
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
            )
        elif args.command == "report":
            from .experiments.reporting import report

            result = report(args.output_root, args.destination)
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
