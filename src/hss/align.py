"""
Cross-layer cluster alignment for hidden-state clustering.

Aligns cluster centers across consecutive layers using similarity-based bipartite
matching. Supports Hungarian (optimal) and greedy matchers, plus cosine or
euclidean similarity. Used to assign consistent global IDs to clusters across
the layer hierarchy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .distance import cosine_similarity_matrix, euclidean_distance_sq
from .types import AlignmentResult, AlignmentStep, AlignSpec
from .utils import f32


def _hungarian(score: np.ndarray) -> List[Tuple[int, int, float]]:
    """Compute maximum-weight bipartite matching via Hungarian algorithm.

    Args:
        score: [n, m] similarity matrix (higher = better match).

    Returns:
        List of (i, j, score[i,j]) for matched pairs, one per row/column.
    """
    if score.size == 0:
        return []
    ri, cj = linear_sum_assignment(-score.astype(np.float64))
    return [(int(i), int(j), float(score[i, j])) for i, j in zip(ri, cj)]


def _greedy(score: np.ndarray) -> List[Tuple[int, int, float]]:
    """Greedy bipartite matching: iterate over pairs by descending score.

    Each pair is chosen greedily; once (i, j) is matched, neither i nor j
    is available for later pairs. Faster than Hungarian but suboptimal.

    Args:
        score: [n, m] similarity matrix (higher = better match).

    Returns:
        List of (i, j, score[i,j]) for matched pairs.
    """
    n, m = score.shape
    if n == 0 or m == 0:
        return []
    flat: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            flat.append((float(score[i, j]), i, j))
    flat.sort(reverse=True, key=lambda x: x[0])
    used_i: set = set()
    used_j: set = set()
    pairs: List[Tuple[int, int, float]] = []
    for s, i, j in flat:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j, s))
        if len(used_i) == n or len(used_j) == m:
            break
    return pairs


# Registry of matcher functions (hungarian: optimal; greedy: faster, suboptimal)
_MATCHERS = {"hungarian": _hungarian, "greedy": _greedy}

# Supported similarity metrics for center matching
_SIMILARITIES = {"cosine", "euclidean"}


def _similarity_matrix(
    A: np.ndarray, B: np.ndarray, similarity: str
) -> np.ndarray:
    """Compute pairwise similarity between rows of A and B.

    Args:
        A: [n, D] center vectors (previous layer).
        B: [m, D] center vectors (current layer).
        similarity: "cosine" | "euclidean".

    Returns:
        [n, m] similarity matrix; higher = more similar. For euclidean,
        uses 1 / (1 + distance) to convert distance to similarity.
    """
    if similarity == "cosine":
        return cosine_similarity_matrix(A, B)
    if similarity == "euclidean":
        d2 = euclidean_distance_sq(A, B)
        return (1.0 / (1.0 + np.sqrt(d2))).astype(np.float32, copy=False)
    raise ValueError(f"Unknown similarity: {similarity}")


def match_layer_pair(
    prev_centers: np.ndarray,
    curr_centers: np.ndarray,
    *,
    spec: AlignSpec,
) -> Tuple[Dict[int, int], np.ndarray]:
    """Match cluster centers between two consecutive layers.

    Builds a similarity matrix, runs the chosen matcher (hungarian/greedy),
    and filters pairs below threshold. Returns mapping from current-layer
    indices to previous-layer indices for matched pairs only.

    Args:
        prev_centers: [k_prev, D] center vectors from previous layer.
        curr_centers: [k_curr, D] center vectors from current layer.
        spec: Alignment spec (similarity, method, threshold, allow_negative).

    Returns:
        mapping: {j_curr -> i_prev} for matched pairs meeting threshold.
        sim: [k_prev, k_curr] similarity matrix.
    """
    if spec.similarity not in _SIMILARITIES:
        raise ValueError(
            f"Unknown similarity '{spec.similarity}', "
            f"expected one of {sorted(_SIMILARITIES)}"
        )
    prev_centers = f32(prev_centers)
    curr_centers = f32(curr_centers)
    kp, kc = prev_centers.shape[0], curr_centers.shape[0]
    if kp == 0 or kc == 0:
        return {}, np.empty((0, 0), dtype=np.float32)
    sim = _similarity_matrix(prev_centers, curr_centers, spec.similarity)
    score = sim.astype(np.float64, copy=False)
    if not spec.allow_negative:
        np.maximum(score, 0.0, out=score)
    matcher = _MATCHERS.get(spec.method)
    if matcher is None:
        raise ValueError(f"Unknown alignment method: {spec.method}")
    pairs = matcher(score)
    mapping: Dict[int, int] = {}
    for i_prev, j_curr, _ in pairs:
        if float(sim[i_prev, j_curr]) >= spec.threshold:
            mapping[j_curr] = i_prev
    return mapping, sim


def align_layers(
    centers_by_layer: List[np.ndarray],
    *,
    layers: Optional[List[int]] = None,
    spec: Optional[AlignSpec] = None,
) -> AlignmentResult:
    """Sequentially align cluster centers across all layers.

    The first layer's clusters keep local IDs as global IDs. For each
    subsequent layer, cluster centers are matched to the previous layer
    via similarity (cosine or euclidean) and bipartite matching. Matched
    clusters inherit the previous global ID; unmatched clusters receive
    new global IDs (births). Global IDs no longer matched are deaths.

    Args:
        centers_by_layer: List of [k_l, D] arrays, one per layer.
        layers: Optional layer indices (default 0..L-1).
        spec: Alignment spec; defaults to AlignSpec().

    Returns:
        AlignmentResult with layers, local_to_global mapping, steps
        (matched/births/deaths per pair), and n_global_states.
    """
    if spec is None:
        spec = AlignSpec()
    L = len(centers_by_layer)
    if layers is None:
        layers = list(range(L))
    if len(layers) != L:
        raise ValueError("layers length != centers_by_layer length")
    if L == 0:
        return AlignmentResult(
            layers=[], local_to_global=[], steps=[], n_global_states=0
        )

    local_to_global: List[np.ndarray] = []
    steps: List[AlignmentStep] = []
    # First layer: local IDs 0..k0-1 become global IDs 0..k0-1
    k0 = centers_by_layer[0].shape[0]
    g0 = np.arange(k0, dtype=np.int32)
    local_to_global.append(g0)
    next_gid = k0
    alive_prev = set(g0.tolist())

    # Sequentially align each layer to the previous one
    for idx in range(1, L):
        prev = f32(centers_by_layer[idx - 1])
        cur = f32(centers_by_layer[idx])
        kp, kc = prev.shape[0], cur.shape[0]
        mapping, sim = match_layer_pair(prev, cur, spec=spec)
        prev_g = local_to_global[-1]  # [k_prev] global ID per prev local index
        cur_g = np.full(kc, -1, dtype=np.int32)  # -1 = unmatched (will be birth)
        matched_prev_gids: set = set()
        matched_pairs: List[Tuple[int, int, float]] = []
        # Propagate global IDs from matched prev clusters to current clusters
        for j_cur, i_prev in mapping.items():
            gid = int(prev_g[i_prev])
            cur_g[j_cur] = gid
            matched_prev_gids.add(gid)
            matched_pairs.append(
                (i_prev, j_cur, float(sim[i_prev, j_cur]))
            )
        births: List[int] = []
        for j in range(kc):
            if cur_g[j] < 0:
                cur_g[j] = next_gid
                births.append(next_gid)
                next_gid += 1
        alive_cur = set(cur_g.tolist())
        # Deaths: global IDs present in prev layer but not matched this step
        deaths = sorted(alive_prev - matched_prev_gids)
        steps.append(
            AlignmentStep(
                layer_from=layers[idx - 1],
                layer_to=layers[idx],
                matched=matched_pairs,
                births=births,
                deaths=deaths,
            )
        )
        local_to_global.append(cur_g)
        alive_prev = alive_cur

    return AlignmentResult(
        layers=list(layers),
        local_to_global=local_to_global,
        steps=steps,
        n_global_states=int(next_gid),
    )