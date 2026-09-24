"""Resident-input, batched float64 MFA EM and safeguarded SQUAREM.

The statistical model is unchanged. SQUAREM is an optional, separately recorded
optimizer: extrapolated parameters must be feasible and improve on two plain
EM updates, otherwise those updates are retained. Convergence is checked using
an ordinary EM likelihood increment, never a rejected extrapolation.
"""
import time
import numpy as np
from .mfa import MFAModel


class BatchedMFAEM:
    def __init__(self, X, *, device="cuda:0", chunk_size=1024, component_batch=8,
                 reg_covar=1e-6):
        import torch
        self.t = torch
        self.device = device
        self.X = torch.as_tensor(np.asarray(X).copy(), dtype=torch.float64, device=device)
        if self.X.ndim != 2 or not bool(torch.isfinite(self.X).all()):
            raise ValueError("Expected finite N by D data")
        if min(chunk_size, component_batch) < 1:
            raise ValueError("Batch sizes must be positive")
        self.X2 = self.X.square()
        self.n, self.d = self.X.shape
        self.batch, self.tile, self.reg = chunk_size, component_batch, reg_covar
        self.evaluations = 0

    def parameters(self, model):
        return tuple(self.t.as_tensor(a.copy(), dtype=self.t.float64, device=self.device)
                     for a in (model.weights_, model.means_, model.loadings_, model.noise_))

    def model(self, p, history, converged=False):
        return MFAModel(*(a.detach().cpu().numpy().copy() for a in p),
                        reg_covar=self.reg, history_=list(history), converged_=converged,
                        backend_="gpu_batched" if str(self.device).startswith("cuda") else "cpu_batched",
                        chunk_size=self.batch)

    def evaluate(self, p, *, update=True, probabilities=False):
        t = self.t
        w, mu, W, psi = p
        k, d, q = W.shape
        inv = psi.reciprocal()
        eye = t.eye(q, dtype=t.float64, device=self.device).expand(k, q, q)
        M = eye + W.transpose(1, 2) @ (inv[..., None] * W)
        chol = t.linalg.cholesky(M)
        C = t.cholesky_inverse(chol)
        P = C @ (W.transpose(1, 2) * inv[:, None, :])
        logdet = psi.log().sum(1) + 2 * chol.diagonal(dim1=1, dim2=2).log().sum(1)
        const = w.log() - .5 * (d * np.log(2 * np.pi) + logdet)
        zeros = lambda *shape: t.zeros(shape, dtype=t.float64, device=self.device)
        nk, R, T, xx = zeros(k), zeros(k, q+1, q+1), zeros(k, d, q+1), zeros(k, d)
        total = zeros()
        probs = []
        for start in range(0, self.n, self.batch):
            X = self.X[start:start+self.batch]
            log, ez_all = zeros(len(X), k), zeros(k, len(X), q)
            for j in range(0, k, self.tile):
                sl = slice(j, min(j+self.tile, k))
                delta = X[None, :, :] - mu[sl, None, :]
                ez = delta @ P[sl].transpose(1, 2)
                low = (delta * inv[sl, None, :]) @ W[sl]
                mahal = (delta.square() * inv[sl, None, :]).sum(2) - (ez * low).sum(2)
                log[:, sl] = (const[sl, None] - .5 * mahal).T
                ez_all[sl] = ez
            normalizer = t.logsumexp(log, dim=1)
            resp = (log - normalizer[:, None]).exp()
            total += normalizer.sum()
            if probabilities:
                probs.append(resp.detach().cpu().numpy())
            if update:
                rt = resp.T
                count = rt.sum(1)
                rz = rt[..., None] * ez_all
                nk += count
                R[:, 0, 0] += count
                R[:, 0, 1:] += rz.sum(1)
                R[:, 1:, 0] += rz.sum(1)
                R[:, 1:, 1:] += ez_all.transpose(1, 2) @ rz + count[:, None, None] * C
                T[:, :, 0] += rt @ X
                T[:, :, 1:] += X.T[None, :, :] @ rz
                xx += rt @ self.X2[start:start+len(X)]
        ll = float(total.cpu()) / self.n
        self.evaluations += 1
        if not np.isfinite(ll):
            raise FloatingPointError("Nonfinite likelihood")
        new = None
        if update:
            if bool((nk < 1e-8).any()):
                raise FloatingPointError("Collapsed component")
            A = t.linalg.solve(R, T.transpose(1, 2)).transpose(1, 2)
            new = (nk/self.n, A[:, :, 0], A[:, :, 1:],
                   ((xx - (A*T).sum(2))/nk[:, None]).clamp_min(self.reg))
        return ll, new, np.concatenate(probs) if probabilities else None

    def feasible(self, p):
        t = self.t
        return (all(bool(t.isfinite(a).all()) for a in p)
                and bool((p[0] > 0).all()) and bool((p[3] >= self.reg).all()))

    def fit(self, initial, *, max_steps=2000, tol=1e-5, accelerator="none",
            deadline=None, checkpoint=None, checkpoint_seconds=60):
        if accelerator not in ("none", "squarem") or max_steps < 1 or tol < 0:
            raise ValueError("Invalid optimizer settings")
        p = self.parameters(initial)
        history = []
        accepted, rejected, converged, last_save = 0, 0, False, time.time()
        ll, p1, _ = self.evaluate(p)
        history.append(ll)
        steps = 0
        # max_steps counts plain EM maps, including the two maps in SQUAREM.
        while steps < max_steps and (deadline is None or time.time() < deadline):
            ll1, p2, _ = self.evaluate(p1)
            steps += 1
            delta = ll1 - ll
            if delta < -1e-6 * max(1., abs(ll)):
                raise FloatingPointError(f"EM likelihood decreased: {delta}")
            if abs(delta) < tol:
                p, ll, converged = p1, ll1, True
                history.append(ll)
                break
            if accelerator == "squarem" and steps < max_steps:
                ll2, p3, _ = self.evaluate(p2)
                steps += 1
                r = tuple(b-a for a,b in zip(p,p1))
                v = tuple(c-2*b+a for a,b,c in zip(p,p1,p2))
                nr = sum(float((a*a).sum().cpu()) for a in r)
                nv = sum(float((a*a).sum().cpu()) for a in v)
                alpha = -float(np.clip(np.sqrt(nr/max(nv,1e-300)), 1., 32.))
                chosen, score, following = p2, ll2, p3
                improved = False
                for _ in range(4):
                    trial = tuple(a-2*alpha*b+alpha*alpha*c for a,b,c in zip(p,r,v))
                    if self.feasible(trial):
                        trial = (trial[0]/trial[0].sum(), *trial[1:])
                        try:
                            value, after, _ = self.evaluate(trial)
                            if value >= ll2:
                                chosen, score, following = trial, value, after
                                improved = True
                                break
                        except (RuntimeError, FloatingPointError):
                            pass
                    alpha = (alpha-1)/2
                accepted += int(improved)
                rejected += int(not improved)
                p, ll, p1 = chosen, score, following
            else:
                p, ll, p1 = p1, ll1, p2
            history.append(ll)
            if checkpoint and time.time()-last_save >= checkpoint_seconds:
                checkpoint(self.model(p, history), dict(steps=steps, evaluations=self.evaluations))
                last_save = time.time()
        model = self.model(p, history, converged)
        audit = dict(optimizer=accelerator, steps=steps, evaluations=self.evaluations,
                     squarem_accepted=accepted, squarem_rejected=rejected,
                     stopped_by_deadline=deadline is not None and time.time() >= deadline,
                     convergence="absolute ordinary-EM average-log-likelihood increment", tol=tol)
        if checkpoint:
            checkpoint(model, audit)
        return model, audit
