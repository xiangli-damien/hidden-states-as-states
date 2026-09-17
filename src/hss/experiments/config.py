"""Strict TOML/JSON configuration, inheritance and typed dotted overrides."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from ..data import DataSpec
from ..cluster.methods import METHODS


@dataclass
class ClusterConfig:
    method: str = "gmm"
    k: int | None = None
    k_min: int = 2
    k_max: int = 80
    k_values: list[int] | None = None
    rank: int = 8
    covariance_type: str = "diag"
    reg_covar: float = 1e-6
    n_init: int = 2
    max_iter: int = 200
    tol: float = 1e-5
    chunk_size: int = 1024
    batch_size: int = 1024
    backend: str = "cpu"
    device: str = "cuda:0"
    adaptive_reg: bool = False
    adaptive_alpha: float = 0.01
    adaptive_min: float = 1e-12
    init_method: str = "kmeans++"
    parsimony_tolerance: float = 0.02
    assignment: str = "nearest"


@dataclass
class TransformConfig:
    standardize: bool = False
    pca_components: int | None = None
    whiten: bool = False


@dataclass
class AlignmentConfig:
    similarity: str = "cosine"
    method: str = "hungarian"
    threshold: float = 0.6


@dataclass
class EvaluationConfig:
    mode: str = "geometry"
    train_fraction: float = 0.4
    validation_fraction: float = 0.2
    split_seed: int = 42
    alpha: float = 1.0
    far_target: float = 0.1
    far_scope: str = "all_boundaries"
    methods: list[str] = field(
        default_factory=lambda: [
            "HSS-NB",
            "LDA",
            "GaussianNB",
            "Logistic",
            "LinearSVM",
            "MLP",
        ]
    )
    continuous_features: str = "last_layer"
    continuous_standardize: bool = True
    logistic_c: float = 1.0
    svm_c: float = 1.0
    mlp_hidden: list[int] = field(default_factory=lambda: [128, 64])
    max_iter: int = 500
    fit_fraction: float = 1.0
    fixed_map: str | None = None
    fixed_k_map: str | None = None
    positive_label: int = 1
    diagnostics: list[str] = field(default_factory=lambda: ["dispersion", "separation"])


@dataclass
class ExecutionConfig:
    cache_root: str = "./.cache/hss"
    artifact_cache_root: str | None = None
    output_root: str = "./outputs"
    workers: int = 1
    threads_per_worker: int = 1
    memory_gib: float = 32.0
    min_available_gib: float = 8.0
    min_gpu_free_gib: float = 8.0
    task_index: int = 0
    task_count: int = 1
    max_trials: int = 10000
    retry_failed: bool = True


@dataclass
class Experiment:
    name: str = "experiment"
    seed: int = 42
    data: DataSpec = field(default_factory=lambda: DataSpec(paths=[]))
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    grid: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        d = deepcopy(payload)
        for key, typ in [
            ("data", DataSpec),
            ("cluster", ClusterConfig),
            ("transform", TransformConfig),
            ("alignment", AlignmentConfig),
            ("evaluation", EvaluationConfig),
            ("execution", ExecutionConfig),
        ]:
            if key in d:
                d[key] = typ(**d[key])
        result = cls(**d)
        result.validate()
        return result

    def validate(self):
        self.data.validate()
        c, e, x = self.cluster, self.evaluation, self.execution
        if c.method not in METHODS or c.backend not in (
            "cpu",
            "gpu",
            "sklearn",
        ):
            raise ValueError("Invalid cluster method/backend")
        if c.covariance_type not in ("diag", "spherical", "tied", "full"):
            raise ValueError("Invalid GMM covariance type")
        if c.method == "mfa" and c.backend == "sklearn":
            raise ValueError("MFA supports cpu/gpu")
        if c.method in ("kmeans", "minibatch_kmeans") and c.backend == "gpu":
            raise ValueError("KMeans control uses CPU; choose backend=cpu")
        if c.assignment not in ("nearest", "posterior"):
            raise ValueError("assignment must be nearest or posterior")
        if c.k_min < 1 or c.k_max < c.k_min or (c.k is not None and c.k < 1):
            raise ValueError("Invalid component range")
        if c.k_values is not None and (
            not c.k_values or any(k < 1 for k in c.k_values)
        ):
            raise ValueError("k_values must be positive")
        if (
            min(c.n_init, c.max_iter, c.chunk_size, c.batch_size) < 1
            or c.rank < 0
            or c.reg_covar <= 0
            or c.tol < 0
        ):
            raise ValueError("Invalid clustering parameters")
        if c.parsimony_tolerance < 0 or self.seed < 0:
            raise ValueError("Invalid tolerance or seed")
        if self.alignment.similarity not in (
            "cosine",
            "euclidean",
        ) or self.alignment.method not in ("hungarian", "greedy"):
            raise ValueError("Invalid alignment")
        if e.mode not in ("geometry", "prediction", "monitoring", "global_control"):
            raise ValueError("Invalid evaluation.mode")
        if e.mode == "prediction" and self.data.representation != "prompt_last":
            raise ValueError(
                "Before-generation prediction requires prompt_last, avoiding response leakage"
            )
        if e.mode == "monitoring" and self.data.representation != "prefix":
            raise ValueError("Sentence monitoring requires prefix representations")
        if (
            not 0 < e.train_fraction < 1
            or not 0 < e.validation_fraction < 1
            or not 0 < e.fit_fraction <= 1
        ):
            raise ValueError("Invalid split/subsample fraction")
        if e.mode == "monitoring" and e.train_fraction + e.validation_fraction >= 1:
            raise ValueError(
                "Monitoring needs disjoint train, validation and test groups"
            )
        if e.alpha <= 0 or not 0 <= e.far_target < 1 or e.positive_label not in (0, 1):
            raise ValueError("Invalid smoothing, FAR target, or label polarity")
        if e.far_scope not in ("all_boundaries", "nonfinal"):
            raise ValueError("far_scope must be all_boundaries or nonfinal")
        if e.continuous_features not in ("last_layer", "all_layers"):
            raise ValueError("Invalid continuous feature mode")
        methods = {
            "HSS-NB",
            "HSS-Markov",
            "LDA",
            "GaussianNB",
            "Logistic",
            "LinearSVM",
            "MLP",
        }
        if set(e.diagnostics) - {"dispersion", "separation", "effective_rank"}:
            raise ValueError("Unknown diagnostic")
        if e.fixed_map and e.mode in ("prediction", "monitoring"):
            raise ValueError(
                "Supervised protocols fit their map on their own training split; fixed_map is geometry-only"
            )
        if (
            self.alignment.similarity == "cosine"
            and not -1 <= self.alignment.threshold <= 1
        ):
            raise ValueError("Cosine alignment threshold must lie in [-1, 1]")
        if set(e.methods) - methods:
            raise ValueError(f"Unknown predictor methods: {set(e.methods) - methods}")
        if (
            min(x.workers, x.threads_per_worker, x.task_count, x.max_trials) < 1
            or not 0 <= x.task_index < x.task_count
        ):
            raise ValueError("Invalid execution limits")
        if x.memory_gib <= 0 or min(x.min_available_gib, x.min_gpu_free_gib) < 0:
            raise ValueError("Invalid resource budgets")
        if (
            self.transform.pca_components is not None
            and self.transform.pca_components < 1
        ):
            raise ValueError("PCA dimension must be positive")


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def set_value(payload, key, value):
    parts = key.split(".")
    node = payload
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f"Unknown config key {key}")
        node = node[part]
    if parts[-1] not in node:
        raise ValueError(f"Unknown config key {key}")
    node[parts[-1]] = value


def load(path, overrides=(), _seen=None):
    path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError("Cyclic config inheritance")
    seen.add(path)
    raw = path.read_text()
    payload = json.loads(raw) if path.suffix == ".json" else tomllib.loads(raw)
    parent = payload.pop("extends", None)
    base = (
        load(path.parent / parent, _seen=seen).to_dict()
        if parent
        else Experiment().to_dict()
    )
    result = merge(base, payload)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError("Overrides use dotted.key=JSON_VALUE")
        set_value(result, key, json.loads(value))
    for key in ("cache_root", "artifact_cache_root", "output_root"):
        if result["execution"][key] is not None:
            result["execution"][key] = os.path.expandvars(
                os.path.expanduser(result["execution"][key])
            )
    for key in ("fixed_map", "fixed_k_map"):
        if result["evaluation"][key]:
            result["evaluation"][key] = os.path.expandvars(
                os.path.expanduser(result["evaluation"][key])
            )
    result["data"]["paths"] = [
        os.path.expandvars(os.path.expanduser(p)) for p in result["data"]["paths"]
    ]
    return Experiment.from_dict(result)
