"""Internal Model Control (IMC) tuning, FOPDT plant.

Rivera-Morari / Skogestad ideal PID parametrisation:

    Kp = (2*tau + L) / (K * (2*lambda + L))
    Ti = tau + L/2
    Td = (tau * L) / (2*tau + L)

`lambda` is the desired closed-loop time constant. Smaller -> faster
but more aggressive. Default: max(0.1*tau, 0.8*L).
"""
from __future__ import annotations

from typing import Callable

from .base import Tuner, TunerResult
from .fopdt_fit import fit_fopdt


class ImcTuner(Tuner):
    name = 'imc'

    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        c = self.config
        probe, metrics, t_arr, y_arr = self.probe_response(evaluate, log)
        m = fit_fopdt(t_arr, y_arr, c.target_delta)
        log(f'[imc] FOPDT K={m.K:.3f} tau={m.tau:.3f}s L={m.L:.4f}s')

        if not m.valid:
            log('[imc] FOPDT fit failed; returning probe.')
            return TunerResult(gains=probe, cost=metrics.cost,
                               history=[{'probe': probe}],
                               notes='FOPDT fit failed (response too small)')

        lam = float(c.extra.get('imc_lambda',
                                max(0.1 * m.tau, 0.8 * max(m.L, 1e-3))))
        Kp = (2.0 * m.tau + m.L) / (m.K * (2.0 * lam + m.L))
        Ti = m.tau + m.L / 2.0
        Td = (m.tau * m.L) / (2.0 * m.tau + m.L) if (2.0 * m.tau + m.L) > 0 else 0.0
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        cand = self._clip_gains(Kp, Ki, Kd)
        m_final, *_ = evaluate(cand)
        log(f'[imc] lambda={lam:.3f}s -> {cand} cost={m_final.cost:.3f}')

        return TunerResult(
            gains=cand,
            cost=m_final.cost,
            history=[{'probe': probe, 'K': m.K, 'tau': m.tau, 'L': m.L,
                      'lambda': lam},
                     {'candidate': cand, 'cost': m_final.cost,
                      'overshoot': m_final.overshoot_pct,
                      'settling': m_final.settling_time_s}],
            notes=f'lambda={lam:.3f} K={m.K:.3f} tau={m.tau:.3f} L={m.L:.4f}',
        )
