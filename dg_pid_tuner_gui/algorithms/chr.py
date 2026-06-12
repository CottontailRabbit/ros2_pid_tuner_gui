"""Chien-Hrones-Reswick (CHR) tuning, FOPDT plant.

Default mode: setpoint regulation, 0% overshoot (aperiodic).
    Kp = 0.6 * tau / (K * L)
    Ti = tau
    Td = 0.5 * L

Other modes via TunerConfig.extra['chr_mode']:
    'setpoint_0', 'setpoint_20', 'disturbance_0', 'disturbance_20'
"""
from __future__ import annotations

from typing import Callable

from .base import Tuner, TunerResult
from .fopdt_fit import fit_fopdt


_RULES = {
    # mode: (Kp_coef, Ti_factor_kind, Ti_coef, Td_coef)
    # Ti_factor_kind: 'tau' or 'L'
    'setpoint_0':     (0.60, 'tau', 1.000, 0.500),
    'setpoint_20':    (0.95, 'tau', 1.357, 0.473),
    'disturbance_0':  (0.95, 'L',   2.400, 0.420),
    'disturbance_20': (1.20, 'L',   2.000, 0.420),
}


class ChrTuner(Tuner):
    name = 'chr'

    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        c = self.config
        mode = str(c.extra.get('chr_mode', 'setpoint_0'))
        if mode not in _RULES:
            log(f'[chr] unknown mode {mode}; using setpoint_0')
            mode = 'setpoint_0'
        Kp_c, Ti_kind, Ti_c, Td_c = _RULES[mode]

        probe, metrics, t_arr, y_arr = self.probe_response(evaluate, log)
        m = fit_fopdt(t_arr, y_arr, c.target_delta)
        log(f'[chr/{mode}] FOPDT K={m.K:.3f} tau={m.tau:.3f}s L={m.L:.4f}s')

        if not m.valid or m.L <= 0:
            log('[chr] FOPDT fit failed; returning probe.')
            return TunerResult(gains=probe, cost=metrics.cost,
                               history=[{'probe': probe, 'mode': mode}],
                               notes='FOPDT fit failed (response too small)')

        Kp = Kp_c * m.tau / (m.K * m.L)
        Ti = (Ti_c * m.tau) if Ti_kind == 'tau' else (Ti_c * m.L)
        Td = Td_c * m.L
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        cand = self._clip_gains(Kp, Ki, Kd)
        m_final, *_ = evaluate(cand)
        log(f'[chr/{mode}] candidate {cand} cost={m_final.cost:.3f}')

        return TunerResult(
            gains=cand,
            cost=m_final.cost,
            history=[{'probe': probe, 'mode': mode, 'K': m.K, 'tau': m.tau,
                      'L': m.L},
                     {'candidate': cand, 'cost': m_final.cost,
                      'overshoot': m_final.overshoot_pct,
                      'settling': m_final.settling_time_s}],
            notes=f'mode={mode} K={m.K:.3f} tau={m.tau:.3f} L={m.L:.4f}',
        )
