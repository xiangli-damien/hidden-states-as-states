"""Finish the authorized finite study and rebuild the portable report per stage."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from hss.experiments.artifacts import save_json, lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = Path(args.study).resolve()
    stages = ["core", "prediction", "reliability", "monitoring"]

    def render():
        # Fresh interpreter: a long fit may outlive a plotting-only code update.
        subprocess.run(
            [
                sys.executable,
                "scripts/render_math_review.py",
                "--study",
                str(root),
                "--destination",
                args.report,
            ],
            check=True,
        )

    with lock(root / "pipeline.lock"):
        status = {"status": "running", "stages": {s: "pending" for s in stages}}
        try:
            for stage in stages:
                status["stages"][stage] = "running"
                status["updated_at"] = time.time()
                save_json(root / "pipeline.json", status)
                command = [
                    sys.executable,
                    "-u",
                    "scripts/run_math_analysis.py",
                    "--directory",
                    str(root),
                    "--workers",
                    str(args.workers),
                    "--stage",
                    stage,
                ]
                with (root / f"pipeline-{stage}.log").open("a") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                status["stages"][stage] = "complete"
                save_json(root / "pipeline.json", status)
                render()
        except Exception as exc:
            status.update(status="failed", error=repr(exc), updated_at=time.time())
            save_json(root / "pipeline.json", status)
            raise
        status.update(status="complete", updated_at=time.time())
        save_json(root / "pipeline.json", status)
        render()
