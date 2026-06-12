"""Ziegler-Nichols open-loop (reaction-curve) tuning.

PID rules from a FOPDT identification:
    Kp = 1.2 * tau / (K * L)
    Ti = 2 * L
    Td = 0.5 * L
"""
from __future__ import annotations

from typing import Callable

from .base import Tuner, TunerResult
from .fopdt_fit import fit_fopdt


class ZieglerNicholsTuner(Tuner):
    name = 'ziegler-nichols'

    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        c = self.config
        probe, metrics, t_arr, y_arr = self.probe_response(evaluate, log)
        m = fit_fopdt(t_arr, y_arr, c.target_delta)
        log(f'[zn-open] FOPDT K={m.K:.3f} tau={m.tau:.3f}s L={m.L:.4f}s')

        if not m.valid or m.L <= 0:
            log('[zn-open] FOPDT fit failed; returning probe.')
            return TunerResult(gains=probe, cost=metrics.cost,
                               history=[{'probe': probe}],
                               notes='FOPDT fit failed (response too small)')

        Kp = 1.2 * m.tau / (m.K * m.L)
        Ti = 2.0 * m.L
        Td = 0.5 * m.L
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        cand = self._clip_gains(Kp, Ki, Kd)
        m_final, *_ = evaluate(cand)
        log(f'[zn-open] candidate {cand} cost={m_final.cost:.3f}')

        return TunerResult(
            gains=cand,
            cost=m_final.cost,
            history=[{'probe': probe, 'K': m.K, 'tau': m.tau, 'L': m.L},
                     {'candidate': cand, 'cost': m_final.cost,
                      'overshoot': m_final.overshoot_pct,
                      'settling': m_final.settling_time_s}],
            notes=f'K={m.K:.3f} tau={m.tau:.3f} L={m.L:.4f}',
        )
