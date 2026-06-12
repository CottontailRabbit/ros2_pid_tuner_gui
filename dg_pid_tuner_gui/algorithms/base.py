"""Tuner abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class TunerConfig:
    joint: str
    target_delta: float = 0.2            # rad step from center (signed)
    duration: float = 2.0                # capture window per trial (s)
    settle_time: float = 0.7             # hold-at-center settle before each step (s)
    max_iter: int = 20                   # for iterative methods
    p_bounds: tuple[float, float] = (0.05, 8.0)
    i_bounds: tuple[float, float] = (0.0, 1.5)
    d_bounds: tuple[float, float] = (0.0, 0.3)
    weights: tuple[float, float, float] = (1.0, 1.0, 5.0)  # overshoot, settling, sse
    settle_band: float = 0.02
    safety_margin: float = 0.2           # fraction of joint range trimmed off both ends
    divergence_penalty: float = 50.0
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class TunerResult:
    gains: dict[str, float]
    cost: float
    history: list[dict] = field(default_factory=list)
    notes: str = ''


class Tuner(ABC):
    """Abstract base for an autotuning algorithm.

    The `evaluate` callable is provided by the GUI/backend: it receives a
    candidate gain dict, applies it to the live controller, runs a step,
    and returns (StepMetrics, t_array, y_array). The tuner only needs to
    decide what to try next.
    """

    name: str = 'base'

    def __init__(self, config: TunerConfig) -> None:
        self.config = config

    @abstractmethod
    def run(
        self,
        evaluate: Callable[[dict[str, float]], tuple],
        log: Callable[[str], None] = lambda _msg: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> TunerResult:
        ...

    def _clip_gains(self, p: float, i: float, d: float) -> dict[str, float]:
        pmin, pmax = self.config.p_bounds
        imin, imax = self.config.i_bounds
        dmin, dmax = self.config.d_bounds
        return {
            'p': float(np.clip(p, pmin, pmax)),
            'i': float(np.clip(i, imin, imax)),
            'd': float(np.clip(d, dmin, dmax)),
            'i_clamp_max': abs(float(np.clip(i, imin, imax))) * 2.0,
            'i_clamp_min': -abs(float(np.clip(i, imin, imax))) * 2.0,
        }

    def probe_response(
        self,
        evaluate,
        log=lambda _msg: None,
        factors: tuple[float, ...] = (0.30, 0.55, 0.80),
    ):
        """Run a P-only probe step, escalating P until the joint actually moves.

        FOPDT-based tuners need a response with non-trivial excursion in order
        to identify K/τ/L. We try `p_max * factor` for each factor in order
        and stop as soon as |max(y) - min(y)| >= 0.3 * |Δ|.
        Returns (gains, metrics, t_array, y_array) of the last (best) attempt.
        """
        delta = abs(self.config.target_delta)
        required = max(0.3 * delta, 0.02)
        pmin, pmax = self.config.p_bounds
        last = None
        for k, f in enumerate(factors):
            kp = max(pmin, pmax * float(f))
            gains = self._clip_gains(kp, 0.0, 0.0)
            metrics, t_arr, y_arr = evaluate(gains)
            excursion = float(np.max(y_arr) - np.min(y_arr)) if y_arr.size else 0.0
            log(f'[probe {k+1}/{len(factors)}] P={kp:.3f}  excursion={excursion:.4f} rad'
                f'  (need ≥ {required:.4f})')
            last = (gains, metrics, t_arr, y_arr)
            if excursion >= required:
                return last
        log('[probe] excursion still small after all attempts; '
            'FOPDT fit may be unreliable.')
        return last
