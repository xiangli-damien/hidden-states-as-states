"""Raw point-cloud diagnostics, independent of clustering and correctness labels."""
import numpy as np
from scipy.stats import spearmanr


def covariance_profile(x, lengths):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < 3:
        raise ValueError('Expected finite observations N x D')
    n, d = x.shape
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / n
    eig, vec = np.linalg.eigh(cov)
    if eig[0] < -1e-10 * max(np.trace(cov), 1.):
        raise ValueError('Covariance is not numerically PSD')
    eig, vec = np.maximum(eig[::-1], 0.), vec[:, ::-1]
    trace = float(np.trace(cov)); square = float(np.square(cov).sum())
    np.testing.assert_allclose(eig.sum(), trace, rtol=1e-9)
    np.testing.assert_allclose(eig @ eig, square, rtol=1e-9)
    if trace <= 0:
        raise ValueError('Degenerate point cloud')
    frac = eig / trace
    pr = trace**2 / square
    var = np.diag(cov)
    norm = np.linalg.norm(x, axis=1)
    radius2 = np.square(xc).sum(1)
    energy = np.square(x).mean(0)
    z = np.log1p(np.asarray(lengths, dtype=float)); z -= z.mean()
    if z.std() > 0:
        z /= z.std()
        beta = xc.T @ z / n
    else:
        beta = np.zeros(d)
    explained = float(beta @ beta)
    residual_trace = trace - explained
    residual_square = square - 2 * float(beta @ cov @ beta) + explained**2
    valid = var > max(var.max() * 1e-12, 1e-30)
    kurt = (np.square(np.square(xc[:, valid])).mean(0) / np.square(var[valid])) - 3.
    scores = xc @ vec[:, :min(32, d)]
    pc_kurt = np.square(np.square(scores)).mean(0) / np.maximum(eig[:scores.shape[1]]**2, 1e-30) - 3.
    u = x / np.maximum(norm[:, None], 1e-30)
    # Exact mean over all distinct pairs; no O(N^2) pair materialization.
    cosine_mean = (float(np.square(u.sum(0)).sum()) - float(np.square(u).sum())) / (n*(n-1))
    trace_trim = np.sort(radius2)[:-max(1, int(.01*n))].mean()
    positive = frac > 0
    metrics = dict(n=n, dimension=d, norm_median=float(np.median(norm)),
        norm_q05=float(np.quantile(norm,.05)), norm_q95=float(np.quantile(norm,.95)),
        norm_cv=float(norm.std()/max(norm.mean(),1e-30)), mean_energy_fraction=float(mean@mean/energy.sum()),
        raw_pair_cosine_mean=cosine_mean, trace=trace, effective_dimension=pr,
        diagonal_effective_dimension=float(trace**2/np.square(var).sum()),
        offdiagonal_covariance_fraction=float(1-np.square(var).sum()/square),
        entropy_rank=float(np.exp(-np.sum(frac[positive]*np.log(frac[positive])))),
        pc1_fraction=float(frac[0]), pc8_fraction=float(frac[:8].sum()), pc16_fraction=float(frac[:16].sum()),
        d90=int(np.searchsorted(np.cumsum(frac),.9)+1), d95=int(np.searchsorted(np.cumsum(frac),.95)+1),
        top_energy_coordinate=int(energy.argmax()), top_energy_fraction=float(energy.max()/energy.sum()),
        top_variance_coordinate=int(var.argmax()), top_variance_fraction=float(var.max()/trace),
        coordinate_kurtosis_median=float(np.median(kurt)), pc1_excess_kurtosis=float(pc_kurt[0]),
        radius_fourth_ratio=float(np.mean(np.square(radius2/trace))/(1+2/pr)),
        outer_1pct_energy_fraction=float(np.sort(radius2)[-max(1,int(.01*n)):].sum()/radius2.sum()),
        trimmed_radius_mean_fraction=float(trace_trim/trace),
        log_length_variance_fraction=float(explained/trace),
        length_residual_effective_dimension=float(residual_trace**2/max(residual_square,1e-30)),
        length_norm_spearman=float(spearmanr(lengths,norm).statistic) if np.std(lengths)>0 else 0.)
    return metrics, dict(mean=mean, eigenvalues=eig, top_vectors=vec[:,:32], pc_scores=scores,
                         norms=norm, radius_squared=radius2, coordinate_variance=var,
                         coordinate_energy=energy, pc_excess_kurtosis=pc_kurt)


def partition_profile(x, labels):
    """Empirical ANOVA decomposition; not a likelihood or semantic test."""
    x = np.asarray(x, float); labels = np.asarray(labels)
    if labels.shape != (len(x),):
        raise ValueError('Labels and rows differ')
    mean = x.mean(0); total = float(np.square(x-mean).sum())
    between, sizes = 0., []
    for k in np.unique(labels):
        part = x[labels==k]; sizes.append(len(part))
        between += len(part)*float(np.square(part.mean(0)-mean).sum())
    p = np.asarray(sizes)/len(x)
    return dict(occupied_clusters=len(sizes), min_occupancy=min(sizes),
                occupancy_entropy=float(-np.sum(p*np.log(p))),
                between_variance_fraction=between/total,
                random_partition_expected_fraction=(len(sizes)-1)/(len(x)-1))


def relative_icl_choices(records, budgets, n, d):
    """Compare score differences, not percentages of an arbitrary score origin."""
    rows = [r for r in records if r['converged'] and np.isfinite(r['icl'])]
    best = min(r['icl'] for r in rows)
    return {str(b): min(r['k'] for r in rows if r['icl']-best <= b*n*d) for b in budgets}
