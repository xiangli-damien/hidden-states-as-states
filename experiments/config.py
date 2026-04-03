from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

@dataclass(frozen=True)
class DataConfig:
    source: str = 'openact'
    run_path: str = ''
    memmap_path: str = ''
    labels_path: str = ''
    reduction: str = 'mean'
    layers: Optional[List[int]] = None
    labels: Optional[List[str]] = None
    only_valid: bool = True
    start_pct: Optional[float] = None
    end_pct: Optional[float] = None
    n_points: Optional[int] = None
    dtype: str = 'float32'
    positive_label_key: str = 'correct'
    unit: str = 'item'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DataConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class ScanConfig:
    k_range: Tuple[int, int] = (2, 40)
    seed: int = 42
    stability_repeats: int = 2
    eval_sample: int = 5000
    icl_mode: str = 'bic_plus_2entropy'
    method: str = 'gmm'
    covariance_type: str = 'diag'
    reg_covar: float = 0.01
    gmm_backend: str = 'auto'
    gmm_adaptive_reg: bool = True
    gmm_adaptive_alpha: float = 0.01
    gmm_adaptive_min: float = 1e-12
    gmm_tol: float = 1e-6
    gmm_chunk_size: int = 4096
    gmm_device: str = 'auto'
    gmm_init_method: str = 'kmeans++'
    gmm_auto_gpu_threshold: int = 50000000
    gmm_spectral_affinity: str = 'nearest_neighbors'
    gmm_spectral_n_neighbors: int = 10
    gmm_spectral_gamma: Optional[float] = None
    batch_size: int = 4096
    n_jobs: int = 1
    parallel_backend: str = 'loky'
    transform_steps: List[Dict[str, Any]] = field(default_factory=lambda: [{'name': 'standardize'}])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['k_range'] = list(self.k_range)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScanConfig:
        payload = dict(d)
        if 'k_range' in payload:
            payload['k_range'] = tuple(payload['k_range'])
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class SelectConfig:
    strategy: str = 'icl_parsimonious'
    relative_pct: float = 0.02
    stability_min: float = 0.3
    target_avg_k: float = 8.0
    candidate_pcts: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SelectConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class DiscretizeConfig:
    seed: int = 42
    method: str = 'gmm'
    covariance_type: str = 'diag'
    reg_covar: float = 0.01
    gmm_backend: str = 'auto'
    gmm_adaptive_reg: bool = True
    gmm_adaptive_alpha: float = 0.01
    gmm_adaptive_min: float = 1e-12
    gmm_tol: float = 1e-6
    gmm_chunk_size: int = 4096
    gmm_device: str = 'auto'
    gmm_init_method: str = 'kmeans++'
    gmm_auto_gpu_threshold: int = 50000000
    gmm_spectral_affinity: str = 'nearest_neighbors'
    gmm_spectral_n_neighbors: int = 10
    gmm_spectral_gamma: Optional[float] = None
    transform_steps: List[Dict[str, Any]] = field(default_factory=lambda: [{'name': 'standardize'}])
    alignment_similarity: str = 'cosine'
    alignment_method: str = 'hungarian'
    alignment_threshold: float = 0.0
    batch_size_fit: int = 4096
    batch_size_predict: int = 8192
    n_jobs: int = 1
    parallel_backend: str = 'threading'
    parsimony_tolerance: float = 0.05
    icl_mode: str = 'bic_plus_2entropy'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DiscretizeConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class StabilityConfig:
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    subsample_fracs: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0])
    kmax_values: List[int] = field(default_factory=lambda: [20, 40, 60, 80])
    n_random_baselines: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StabilityConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class PredictionConfig:
    n_splits: int = 5
    seed: int = 42
    alpha: float = 1.0
    methods: List[str] = field(default_factory=lambda: ['HSS-NB', 'HSS-Markov', 'LDA', 'GaussianNB', 'Logistic', 'LinearSVM', 'MLP'])
    mlp_hidden: Tuple[int, ...] = (128, 64)
    mlp_max_iter: int = 500
    continuous_feature_mode: str = 'last_layer'
    continuous_standardize: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['mlp_hidden'] = list(self.mlp_hidden)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictionConfig:
        payload = dict(d)
        if 'mlp_hidden' in payload:
            payload['mlp_hidden'] = tuple(payload['mlp_hidden'])
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class AblationConfig:
    enabled: List[str] = field(default_factory=lambda: ['clustering_method', 'preprocessing', 'cosine_vs_euclidean', 'fit_vs_hidden', 'threshold_sweep', 'propagation', 'alignment'])
    clustering_methods: List[str] = field(default_factory=lambda: ['gmm', 'kmeans'])
    preprocess_variants: List[str] = field(default_factory=lambda: ['standardize', 'pca128', 'standardize+pca128'])
    threshold_points: int = 50
    drift_strides: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 15, 20])
    seed: int = 42
    n_jobs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AblationConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class VisualizationConfig:
    enabled: List[str] = field(default_factory=lambda: ['dynamics_unlabeled', 'dynamics_absolute', 'dynamics_relative', 'heatmap_delta', 'heatmap_probability', 'information_curve', 'information_bottleneck', 'effective_rank', 'rel_icl_surface', 'trend_stability', 'centroid_persistence', 'tolerance_sweep', 'prediction_auroc', 'prediction_folds'])
    selected_layers: Optional[List[int]] = None
    figure_formats: List[str] = field(default_factory=lambda: ['pdf', 'png'])
    cluster_order: str = 'mean_delta'
    figsize_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VisualizationConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class ExecutionConfig:
    stages: List[str] = field(default_factory=lambda: ['base', 'preliminary', 'prediction', 'stability', 'ablation', 'visualization'])
    log_level: str = 'INFO'
    save_yaml: bool = True
    save_json: bool = True
    save_figures: bool = True
    overwrite: bool = False
    quiet: bool = False
    experiment_jobs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExecutionConfig:
        return cls(**{k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__})

@dataclass(frozen=True)
class ExperimentConfig:
    name: str = 'default'
    data: DataConfig = field(default_factory=DataConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    discretize: DiscretizeConfig = field(default_factory=DiscretizeConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output_dir: str = './results'

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'data': self.data.to_dict(), 'scan': self.scan.to_dict(), 'select': self.select.to_dict(), 'discretize': self.discretize.to_dict(), 'stability': self.stability.to_dict(), 'prediction': self.prediction.to_dict(), 'ablation': self.ablation.to_dict(), 'visualization': self.visualization.to_dict(), 'execution': self.execution.to_dict(), 'output_dir': self.output_dir}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExperimentConfig:
        return cls(name=d.get('name', 'default'), data=DataConfig.from_dict(d.get('data', {})), scan=ScanConfig.from_dict(d.get('scan', {})), select=SelectConfig.from_dict(d.get('select', {})), discretize=DiscretizeConfig.from_dict(d.get('discretize', {})), stability=StabilityConfig.from_dict(d.get('stability', {})), prediction=PredictionConfig.from_dict(d.get('prediction', {})), ablation=AblationConfig.from_dict(d.get('ablation', {})), visualization=VisualizationConfig.from_dict(d.get('visualization', {})), execution=ExecutionConfig.from_dict(d.get('execution', {})), output_dir=d.get('output_dir', './results'))

def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    if p.suffix.lower() in {'.yaml', '.yml'}:
        data = yaml.safe_load(text) or {}
    else:
        import json
        data = json.loads(text)
    return ExperimentConfig.from_dict(data)

def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {'.yaml', '.yml'}:
        p.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding='utf-8')
    else:
        import json
        p.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding='utf-8')
    return p
