"""Step-response metric calculation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class StepMetrics:
    overshoot_pct: float        # %
    settling_time_s: float      # 2% band
    rise_time_s: float          # 10%->90%
    steady_state_error: float   # |final - target|
    iae: float                  # integral of absolute error
    cost: float                 # weighted scalar cost

    def as_dict(self) -> dict:
        return {
            'overshoot_pct': self.overshoot_pct,
            'settling_time_s': self.settling_time_s,
            'rise_time_s': self.rise_time_s,
            'steady_state_error': self.steady_state_error,
            'iae': self.iae,
            'cost': self.cost,
        }


def compute_step_metrics(
    t: Sequence[float],
    y: Sequence[float],
    y0: float,
    y_target: float,
    settle_band: float = 0.02,
    weights: tuple[float, float, float] = (1.0, 1.0, 5.0),
    y_safe_min: float | None = None,
    y_safe_max: float | None = None,
    divergence_penalty: float = 50.0,
) -> StepMetrics:
    """
    Compute overshoot/settling/rise/SSE/IAE for a step from y0 -> y_target.

    weights: (w_overshoot, w_settling, w_sse) for the scalar cost.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2 or y.size != t.size:
        return StepMetrics(0.0, 0.0, 0.0, 0.0, 0.0, float('inf'))

    delta = y_target - y0
    if abs(delta) < 1e-12:
        return StepMetrics(0.0, 0.0, 0.0, 0.0, 0.0, float('inf'))

    # Final value: median of last 10% samples (robust)
    n_tail = max(1, int(0.1 * y.size))
    y_final = float(np.median(y[-n_tail:]))
    sse = abs(y_final - y_target)

    # Overshoot
    if delta > 0:
        peak = float(np.max(y))
        overshoot = max(0.0, (peak - y_target) / abs(delta) * 100.0)
    else:
        peak = float(np.min(y))
        overshoot = max(0.0, (y_target - peak) / abs(delta) * 100.0)

    # Rise time (10% -> 90% of step)
    p10 = y0 + 0.1 * delta
    p90 = y0 + 0.9 * delta
    if delta > 0:
        idx_10 = np.argmax(y >= p10) if np.any(y >= p10) else -1
        idx_90 = np.argmax(y >= p90) if np.any(y >= p90) else -1
    else:
        idx_10 = np.argmax(y <= p10) if np.any(y <= p10) else -1
        idx_90 = np.argmax(y <= p90) if np.any(y <= p90) else -1
    rise = float(t[idx_90] - t[idx_10]) if idx_10 >= 0 and idx_90 > idx_10 else float(t[-1] - t[0])

    # Settling time: last time |y - y_target| > settle_band * |delta|
    band = settle_band * abs(delta)
    out_of_band = np.abs(y - y_target) > band
    if not np.any(out_of_band):
        settling = float(t[0] - t[0])
    else:
        last_out = np.where(out_of_band)[0][-1]
        settling = float(t[min(last_out + 1, t.size - 1)] - t[0])

    iae = float(np.trapz(np.abs(y - y_target), t))

    w_o, w_s, w_e = weights
    # Normalize so each term ~O(1) for typical gripper steps
    cost = (
        w_o * (overshoot / 10.0)
        + w_s * settling
        + w_e * (sse / max(abs(delta), 1e-6))
    )

    # Divergence / safety penalty: any sample outside the safe band gets
    # heavily penalised so the optimiser learns to avoid that region.
    if y_safe_min is not None and y.size:
        excess_lo = float(max(0.0, y_safe_min - np.min(y)))
        if excess_lo > 0:
            cost += divergence_penalty * (excess_lo / max(abs(delta), 1e-6) + 1.0)
    if y_safe_max is not None and y.size:
        excess_hi = float(max(0.0, np.max(y) - y_safe_max))
        if excess_hi > 0:
            cost += divergence_penalty * (excess_hi / max(abs(delta), 1e-6) + 1.0)

    return StepMetrics(
        overshoot_pct=overshoot,
        settling_time_s=settling,
        rise_time_s=rise,
        steady_state_error=sse,
        iae=iae,
        cost=cost,
    )
