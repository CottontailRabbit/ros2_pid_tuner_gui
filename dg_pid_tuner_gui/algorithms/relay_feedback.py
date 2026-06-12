"""Relay-feedback (Astrom-Hagglund) autotuner.

Approach:
1. Apply a small relay around the current setpoint by toggling reference
   between (setpoint + h) and (setpoint - h) whenever measured position
   crosses the setpoint. We approximate this offline by stepping with a
   high-P controller and observing the dominant oscillation period Tu
   and amplitude a from the captured response. From those:
       Ku = 4 * h / (pi * a)
   And classic ZN PID rule (less aggressive: Tyreus-Luyben):
       Kp = Ku / 2.2
       Ti = 2.2 * Tu      ->   Ki = Kp / Ti
       Td = Tu / 6.3      ->   Kd = Kp * Td

The implementation uses a single stepped-relay run via the evaluator
because the GUI's `evaluate` already handles set-gains + capture; we
exploit it by running with a deliberately oscillatory P-only controller
and analysing the first response window.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Tuner, TunerConfig, TunerResult


def _detect_period_and_amp(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if t.size < 8:
        return 0.0, 0.0
    y_centered = y - float(np.mean(y))
    # zero crossings (rising)
    sign = np.sign(y_centered)
    crossings = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if crossings.size < 2:
        return 0.0, 0.0
    periods = np.diff(t[crossings])
    Tu = float(np.median(periods))
    amp = float((np.max(y_centered) - np.min(y_centered)) / 2.0)
    return Tu, amp


class RelayFeedbackTuner(Tuner):
    name = 'relay'

    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        c = self.config
        # Push P high enough that the inner system becomes oscillatory.
        history: list[dict] = []
        best: tuple[float, dict] = (float('inf'), {})

        kp_grid = np.linspace(c.p_bounds[0] * 4, c.p_bounds[1] * 0.6, 5)
        Tu_est = 0.0
        amp_est = 0.0
        for kp in kp_grid:
            if is_cancelled():
                break
            gains = self._clip_gains(kp, 0.0, 0.0)
            metrics, t_arr, y_arr = evaluate(gains)
            Tu, amp = _detect_period_and_amp(t_arr, y_arr)
            history.append({'kp_probe': float(kp), 'Tu': Tu, 'amp': amp,
                            'cost': metrics.cost})
            log(f'[relay] probe Kp={kp:.3f}  Tu={Tu:.3f}s amp={amp:.4f}  cost={metrics.cost:.3f}')
            if Tu > 0 and amp > 1e-4:
                Tu_est, amp_est = Tu, amp
                # use this Kp as Ku approx since amplitude > 0 means it oscillates here
                Ku = kp
                # Tyreus-Luyben PID
                Kp = Ku / 2.2
                Ti = 2.2 * Tu
                Td = Tu / 6.3
                Ki = Kp / Ti if Ti > 0 else 0.0
                Kd = Kp * Td
                cand = self._clip_gains(Kp, Ki, Kd)
                m, *_ = evaluate(cand)
                history.append({'candidate': cand, 'cost': m.cost,
                                'overshoot': m.overshoot_pct,
                                'settling': m.settling_time_s})
                log(f'[relay] T-L candidate {cand}  cost={m.cost:.3f}')
                if m.cost < best[0]:
                    best = (m.cost, cand)
                break

        if not best[1]:
            # fallback: lowest-cost probe
            best = (history[-1]['cost'], self._clip_gains(kp_grid[len(history) - 1], 0.0, 0.0))

        return TunerResult(
            gains=best[1],
            cost=best[0],
            history=history,
            notes=f'Tu={Tu_est:.3f}s amp={amp_est:.4f}',
        )
