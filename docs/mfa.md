# Mixture of Factor Analyzers

MFA models component covariance as `W_k W_kᵀ + diag(psi_k)`, with `W_k` of shape `[hidden_dim, rank]` and positive component-specific noise. This captures correlated directions while retaining a small latent dimension.

The implementation follows latent-variable EM with exact posterior latent moments and joint mean/loading updates. It uses Woodbury solves and the determinant lemma, avoids constructing dense hidden-dimension covariance matrices, and streams samples in chunks. CPU NumPy and optional CUDA PyTorch execute the same float64 updates. The output stores ordinary arrays and supports CPU inference without PyTorch.

Reference: [Ghahramani & Hinton, The EM Algorithm for Mixtures of Factor Analyzers, CRG-TR-96-1](https://www.cs.utoronto.ca/~hinton/absps/tr-96-1.pdf).

- `rank=0` gives a diagonal Gaussian mixture.
- `0 <= rank < hidden_dim`; invalid ranks fail instead of being silently clipped.
- `reg_covar` floors diagonal noise; `n_init` controls deterministic KMeans-based restarts.
- Collapsed components and nonfinite likelihoods fail visibly.
- ICL uses `BIC + 2 * responsibility_entropy`; the parameter count subtracts loading rotation redundancy, `K-1 + K*(2D + Dq - q(q-1)/2)`.
- The paper's frozen nearest-centroid assignment remains the default for both GMM and MFA. `assignment="posterior"` is an explicit alternative.
- Per-component history and convergence status are saved in `model.json`. Hitting `max_iter` is a completed fit with `converged=false`, not proof of convergence.

MFA is an extension of the supplied manuscript's diagonal GMM method. Its correlated covariance is more expensive: each iteration scales approximately with `N*K*D*q` plus latent solves, versus `N*K*D` for diagonal GMM. A large rank × K × seed grid can still be costly. Benchmark the intended dimensionality before expanding the grid; CUDA is optional, and small problems can be faster on CPU because device launch overhead dominates.
