"""
Hidden States as States (hss): discretize hidden-state trajectories and analyze them.

Primary API (start here):
  - NumpyProvider, MemmapProvider  — data sources
  - HSSConfig                      — pipeline config
  - discretize                     — main pipeline (transform → cluster → align)
  - scan, ScanConfig               — grid-scan metrics for k selection
  - select_icl_parsimonious, select_sil_stable, select_auto_target_k  — choose k from metrics
  - ArtifactStore                  — save/load results

This revision also exposes RowMetadata so sentence-level and token-level runs can
carry stable row/sample provenance through the core library.
"""
from .version import __version__
from .types import (
    Batch,
    StateProvider,
    TransformStepSpec,
    TransformSpec,
    ClusterSpec,
    AlignSpec,
    HSSConfig,
    RowMetadata,
    LayerResult,
    AlignmentStep,
    AlignmentResult,
    HSSResult,
)
from .provider import NumpyProvider, MemmapProvider, CallableProvider
from .transform import (
    Transform,
    IdentityTransform,
    L2NormTransform,
    StandardizeTransform,
    PCATransform,
    TransformChain,
    build_chain,
    build_chain_from_spec,
)
from .distance import (
    cosine_similarity_matrix,
    euclidean_distance_sq,
    pairwise_distance,
    assign_nearest,
    assign_nearest_chunked,
)
from .cluster import (
    ClusterModel,
    KMeansModel,
    GMMModel,
    fit_cluster,
    rebuild_model,
    select_k_kmeans,
    select_k_gmm,
)
from .align import align_layers, match_layer_pair
from .trajectory import (
    build_global_labels,
    build_trajectory_df,
    count_transitions,
    detect_events,
    EventConfig,
    trajectory_hamming_similarity,
    self_transition_prob,
    active_states_per_layer,
)
from .predict import (
    NaiveBayesClassifier,
    MarkovClassifier,
    evaluate_binary,
)
from .pipeline import discretize, discretize_with_existing_models
from .scan import ScanConfig, scan
from .select import (
    select_auto_target_k,
    select_icl_parsimonious,
    select_sil_stable,
)
from .storage import ArtifactStore
from .splitting import (
    split_indices,
    subsample_indices,
    kfold_indices,
    stratified_split,
)

__primary__ = (
    "NumpyProvider",
    "MemmapProvider",
    "HSSConfig",
    "discretize",
    "scan",
    "ScanConfig",
    "select_icl_parsimonious",
    "select_sil_stable",
    "select_auto_target_k",
    "ArtifactStore",
)

__all__ = [
    "__version__",
    "__primary__",
    "Batch",
    "StateProvider",
    "TransformStepSpec",
    "TransformSpec",
    "ClusterSpec",
    "AlignSpec",
    "HSSConfig",
    "RowMetadata",
    "LayerResult",
    "AlignmentStep",
    "AlignmentResult",
    "HSSResult",
    "NumpyProvider",
    "MemmapProvider",
    "CallableProvider",
    "Transform",
    "IdentityTransform",
    "L2NormTransform",
    "StandardizeTransform",
    "PCATransform",
    "TransformChain",
    "build_chain",
    "build_chain_from_spec",
    "cosine_similarity_matrix",
    "euclidean_distance_sq",
    "pairwise_distance",
    "assign_nearest",
    "assign_nearest_chunked",
    "ClusterModel",
    "KMeansModel",
    "GMMModel",
    "fit_cluster",
    "rebuild_model",
    "select_k_kmeans",
    "select_k_gmm",
    "align_layers",
    "match_layer_pair",
    "build_global_labels",
    "build_trajectory_df",
    "count_transitions",
    "detect_events",
    "EventConfig",
    "trajectory_hamming_similarity",
    "self_transition_prob",
    "active_states_per_layer",
    "NaiveBayesClassifier",
    "MarkovClassifier",
    "evaluate_binary",
    "discretize",
    "discretize_with_existing_models",
    "ScanConfig",
    "scan",
    "select_icl_parsimonious",
    "select_sil_stable",
    "select_auto_target_k",
    "ArtifactStore",
    "split_indices",
    "subsample_indices",
    "kfold_indices",
    "stratified_split",
]
