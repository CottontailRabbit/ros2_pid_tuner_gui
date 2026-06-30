"""First-order plus dead-time model identification from a step response."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FopdtModel:
    K: float    # static gain (Δy / Δu)
    tau: float  # time constant (s)
    L: float    # dead time (s)

    @property
    def valid(self) -> bool:
        return abs(self.K) > 1e-6 and self.tau > 0.0 and self.L >= 0.0


def fit_fopdt(
    t: np.ndarray,
    y: np.ndarray,
    u_step: float,
    min_response_ratio: float = 0.05,
) -> FopdtModel:
    """Two-point method (28.3% / 63.2%) for K, tau, L.

    Rejects fits where the steady-state excursion `|yf - y0|` is smaller
    than `min_response_ratio * |u_step|` — those are dominated by noise
    and produce bogus K/tau/L.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 4 or u_step == 0:
        return FopdtModel(0.0, 0.0, 0.0)

    n_head = max(1, t.size // 20)
    n_tail = max(1, t.size // 10)
    y0 = float(np.mean(y[:n_head]))
    yf = float(np.median(y[-n_tail:]))
    if abs(yf - y0) < min_response_ratio * max(abs(u_step), 1e-3):
        return FopdtModel(0.0, 0.0, 0.0)
    K = (yf - y0) / u_step
    if abs(K) < 1e-3:
        return FopdtModel(0.0, 0.0, 0.0)

    target_28 = y0 + 0.283 * (yf - y0)
    target_63 = y0 + 0.632 * (yf - y0)
    rising = yf >= y0
    if rising:
        i28 = int(np.argmax(y >= target_28)) if np.any(y >= target_28) else -1
        i63 = int(np.argmax(y >= target_63)) if np.any(y >= target_63) else -1
    else:
        i28 = int(np.argmax(y <= target_28)) if np.any(y <= target_28) else -1
        i63 = int(np.argmax(y <= target_63)) if np.any(y <= target_63) else -1
    if i28 < 0 or i63 <= i28:
        return FopdtModel(float(K), 0.0, 0.0)

    t28, t63 = float(t[i28]), float(t[i63])
    tau = 1.5 * (t63 - t28)
    L = max(0.0, t63 - tau)
    return FopdtModel(K=float(K), tau=float(tau), L=float(L))
