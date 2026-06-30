"""Cohen-Coon tuning from a single step response."""
from __future__ import annotations

from typing import Callable

from .base import Tuner, TunerResult
from .fopdt_fit import fit_fopdt


class CohenCoonTuner(Tuner):
    name = 'cohen-coon'

    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        c = self.config
        probe, metrics, t_arr, y_arr = self.probe_response(evaluate, log)
        m = fit_fopdt(t_arr, y_arr, c.target_delta)
        log(f'[cohen-coon] FOPDT K={m.K:.3f} tau={m.tau:.3f}s L={m.L:.4f}s')

        if not m.valid or m.L <= 0:
            log('[cohen-coon] FOPDT fit failed; falling back to probe.')
            return TunerResult(gains=probe, cost=metrics.cost,
                               history=[{'probe': probe, 'cost': metrics.cost}],
                               notes='FOPDT fit failed (response too small)')

        r = m.L / m.tau
        Kp = (1.0 / m.K) * (1.35 / r + 0.25)
        Ti = m.L * (2.5 + 2.0 * r) / (1.0 + 0.6 * r)
        Td = m.L * 0.37 / (1.0 + 0.2 * r)
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        cand = self._clip_gains(Kp, Ki, Kd)
        m_final, *_ = evaluate(cand)
        log(f'[cohen-coon] candidate {cand} cost={m_final.cost:.3f}')

        return TunerResult(
            gains=cand,
            cost=m_final.cost,
            history=[
                {'probe': probe, 'K': m.K, 'tau': m.tau, 'L': m.L,
                 'cost': metrics.cost},
                {'candidate': cand, 'cost': m_final.cost,
                 'overshoot': m_final.overshoot_pct,
                 'settling': m_final.settling_time_s},
            ],
            notes=f'K={m.K:.3f} tau={m.tau:.3f} L={m.L:.4f}',
        )
