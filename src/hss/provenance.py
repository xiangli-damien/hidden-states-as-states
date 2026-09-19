"""Stage-specific source identities: changing a figure never triggers a refit."""

from pathlib import Path


def files_for_stage(stage, root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    shared = [
        "provenance.py",
        "experiments/artifacts.py",
        "types.py",
        "utils.py",
        "distance.py",
    ]
    fit = ["cluster/*.py", "transform/*.py", "experiments/fitting.py"]
    data = ["data/*.py"]
    analysis = [
        "align.py",
        "predict.py",
        "experiments/runner.py",
        "experiments/config.py",
        "experiments/evaluate.py",
        "experiments/diagnostics.py",
        "experiments/resources.py",
        "results/models.py",
        "results/store.py",
    ]
    patterns = {
        "fit": shared + fit,
        "data": shared + data,
        "trial": shared + fit + analysis,
        "figure": shared
        + ["viz/*.py", "analysis/*.py", "results/*.py", "experiments/diagnostics.py"],
    }[stage]
    return sorted(
        {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    )


def stage_version(stage, root=None):
    from .experiments.artifacts import digest, file_digest, runtime_versions

    root = Path(root) if root else Path(__file__).resolve().parent
    packages = {
        "fit": {"python", "numpy", "scipy", "scikit-learn", "torch"},
        "data": {"python", "numpy", "pandas", "zarr", "pyarrow"},
        "trial": {
            "python",
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "torch",
            "pyarrow",
        },
        "figure": {
            "python",
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "matplotlib",
            "pyarrow",
        },
    }[stage]
    return digest(
        {
            "stage": stage,
            "files": {
                str(p.relative_to(root)): file_digest(p)
                for p in files_for_stage(stage, root)
            },
            "runtime": {k: v for k, v in runtime_versions().items() if k in packages},
        }
    )
