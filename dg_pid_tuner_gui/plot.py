"""matplotlib-based step-response plot embedded in PyQt5."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class StepResponsePlot(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel('position [rad]')
        self.ax.grid(True, alpha=0.3)
        self._lines: list = []

    def show_response(
        self,
        t: np.ndarray,
        y: np.ndarray,
        target: float,
        initial: float,
        metrics: dict | None = None,
    ) -> None:
        self.ax.clear()
        self.ax.set_xlabel('time [s]')
        self.ax.set_ylabel('position [rad]')
        self.ax.grid(True, alpha=0.3)
        if t.size == 0:
            self.draw()
            return
        self.ax.plot(t, y, label='response', linewidth=1.4)
        self.ax.axhline(target, color='tab:green', linestyle='--', alpha=0.7,
                        label=f'target ({target:.3f})')
        self.ax.axhline(initial, color='gray', linestyle=':', alpha=0.5,
                        label=f'initial ({initial:.3f})')

        if metrics:
            band = abs(target - initial) * metrics.get('settle_band', 0.02)
            if band > 0:
                self.ax.axhspan(target - band, target + band, color='tab:green',
                                alpha=0.08, label='2% band')
            txt = (
                f"overshoot: {metrics.get('overshoot_pct', 0):.1f}%\n"
                f"settling: {metrics.get('settling_time_s', 0):.2f} s\n"
                f"rise: {metrics.get('rise_time_s', 0):.2f} s\n"
                f"SSE: {metrics.get('steady_state_error', 0):.4f}\n"
                f"cost: {metrics.get('cost', 0):.3f}"
            )
            self.ax.text(0.98, 0.04, txt, transform=self.ax.transAxes,
                         ha='right', va='bottom', fontsize=8,
                         bbox=dict(boxstyle='round', alpha=0.15, facecolor='white'))
        self.ax.legend(loc='best', fontsize=8)
        self.draw()
