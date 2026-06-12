"""Bayesian Optimization tuner (GP + Expected Improvement).

Uses scikit-optimize when available, otherwise falls back to a small
GP implementation built on numpy. The fallback is deliberately simple
but works for the gripper joint problem (3-D search).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .base import Tuner, TunerResult


def _have_skopt() -> bool:
    try:
        import skopt  # noqa: F401
        return True
    except ImportError:
        return False


class BayesianOptTuner(Tuner):
    name = 'bayes-opt'

    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        c = self.config
        bounds = [c.p_bounds, c.i_bounds, c.d_bounds]
        history: list[dict] = []

        def cost_fn(x):
            cand = self._clip_gains(x[0], x[1], x[2])
            metrics, *_ = evaluate(cand)
            history.append({'p': cand['p'], 'i': cand['i'], 'd': cand['d'],
                            'cost': metrics.cost,
                            'overshoot': metrics.overshoot_pct,
                            'settling': metrics.settling_time_s,
                            'sse': metrics.steady_state_error})
            log(f"[bayes] P={cand['p']:.3f} I={cand['i']:.3f} D={cand['d']:.3f}"
                f" -> cost={metrics.cost:.3f} (overshoot={metrics.overshoot_pct:.1f}%,"
                f" settle={metrics.settling_time_s:.2f}s, sse={metrics.steady_state_error:.4f})")
            return float(metrics.cost)

        if _have_skopt():
            from skopt import gp_minimize
            from skopt.space import Real
            space = [
                Real(*c.p_bounds, name='p'),
                Real(*c.i_bounds, name='i'),
                Real(*c.d_bounds, name='d'),
            ]
            res = gp_minimize(
                cost_fn, space, n_calls=c.max_iter,
                n_initial_points=max(4, c.max_iter // 4),
                random_state=c.seed, acq_func='EI', verbose=False,
                callback=[lambda *_: is_cancelled() and (_ for _ in ()).throw(StopIteration)]
                if False else None,
            )
            best_x = list(res.x)
            best_cost = float(res.fun)
            notes = 'skopt.gp_minimize'
        else:
            best_x, best_cost = self._fallback_bo(cost_fn, bounds, c.max_iter, c.seed, is_cancelled)
            notes = 'numpy fallback (no skopt)'

        return TunerResult(
            gains=self._clip_gains(best_x[0], best_x[1], best_x[2]),
            cost=best_cost,
            history=history,
            notes=notes,
        )

    # ---------- numpy fallback BO ------------------------------------

    def _fallback_bo(self, cost_fn, bounds, n_calls, seed, is_cancelled):
        rng = np.random.default_rng(seed)
        n_init = max(4, n_calls // 3)
        X: list[list[float]] = []
        y: list[float] = []

        # Latin hypercube-ish initial design
        for k in range(n_init):
            if is_cancelled():
                break
            x = [
                float(rng.uniform(*bounds[0])),
                float(rng.uniform(*bounds[1])),
                float(rng.uniform(*bounds[2])),
            ]
            X.append(x)
            y.append(cost_fn(x))

        for _ in range(max(0, n_calls - n_init)):
            if is_cancelled():
                break
            xnext = self._propose_ei(np.asarray(X), np.asarray(y), bounds, rng)
            X.append(list(xnext))
            y.append(cost_fn(list(xnext)))

        if not y:
            mid = [(b[0] + b[1]) / 2 for b in bounds]
            return mid, float('inf')
        ibest = int(np.argmin(y))
        return X[ibest], float(y[ibest])

    @staticmethod
    def _kernel(a, b, length=1.0, sigma_f=1.0):
        d = np.sum(a ** 2, axis=1)[:, None] + np.sum(b ** 2, axis=1)[None, :] - 2 * a @ b.T
        return sigma_f ** 2 * np.exp(-0.5 * d / (length ** 2))

    def _propose_ei(self, X, y, bounds, rng, n_candidates=512):
        # normalise inputs to [0,1] for kernel
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        Xn = (X - lo) / (hi - lo + 1e-12)
        yn = (y - np.mean(y)) / (np.std(y) + 1e-9)

        K = self._kernel(Xn, Xn) + 1e-6 * np.eye(len(Xn))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-3 * np.eye(len(Xn)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))

        cand_n = rng.uniform(0.0, 1.0, size=(n_candidates, X.shape[1]))
        Ks = self._kernel(cand_n, Xn)
        mu = Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        var = np.maximum(1e-9, 1.0 - np.sum(v ** 2, axis=0))
        sigma = np.sqrt(var)

        f_best = float(np.min(yn))
        z = (f_best - mu) / sigma
        from math import erf, sqrt
        Phi = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
        phi = (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * z ** 2)
        ei = (f_best - mu) * Phi + sigma * phi
        ei = np.where(sigma < 1e-9, 0.0, ei)
        idx = int(np.argmax(ei))
        return cand_n[idx] * (hi - lo) + lo
