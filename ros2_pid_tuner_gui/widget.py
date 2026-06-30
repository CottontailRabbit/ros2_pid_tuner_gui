"""Main PyQt5 widget for the DG PID tuner."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QPlainTextEdit, QGroupBox, QSplitter, QSizePolicy,
)

from .ros_io import RosBackend
from .controllers import DEFAULT_GAIN_FIELDS
from .plot import StepResponsePlot
from .metrics import compute_step_metrics
from .yaml_io import load_gains, save_gains
from .algorithms import ALGORITHMS
from .algorithms.base import TunerConfig


CENTER_COL_HEADER = 'center [rad]'


class TuneWorker(QObject):
    """Runs an autotuning algorithm in a background thread."""
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    response_captured = pyqtSignal(object, object, float, float, object)

    def __init__(self, backend: RosBackend, joint: str, algorithm_cls,
                 config: TunerConfig, center: float,
                 safe_lo: float, safe_hi: float) -> None:
        super().__init__()
        self.backend = backend
        self.joint = joint
        self.config = config
        self.algorithm = algorithm_cls(config)
        self.center = float(center)
        self.safe_lo = float(safe_lo)
        self.safe_hi = float(safe_hi)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        joint = self.joint
        center = self.center
        delta = float(self.config.target_delta)
        amp = abs(delta)
        try:
            self._history: list = []

            def evaluate(gains: dict[str, float]):
                if self.is_cancelled():
                    raise StopIteration()
                ok = self.backend.set_gains(joint, gains)
                if not ok:
                    self.log.emit(f'set_parameters failed for {gains}')

                # 1. Move to center and let it settle.
                self.backend.hold_position({joint: center})
                end = time.monotonic() + self.config.settle_time
                while time.monotonic() < end:
                    if self.is_cancelled():
                        raise StopIteration()
                    time.sleep(0.05)

                # 2. Alternate step direction to avoid drift.
                direction = 1.0 if (len(self._history) % 2 == 0) else -1.0
                raw_target = center + direction * delta
                tgt = float(np.clip(raw_target, self.safe_lo, self.safe_hi))

                t_arr, y_arr = self.backend.run_step(
                    joint, tgt, duration=self.config.duration)
                y0 = float(y_arr[0]) if y_arr.size else center
                m = compute_step_metrics(
                    t_arr, y_arr, y0=y0, y_target=tgt,
                    settle_band=self.config.settle_band,
                    weights=self.config.weights,
                    y_safe_min=self.safe_lo,
                    y_safe_max=self.safe_hi,
                    divergence_penalty=self.config.divergence_penalty,
                )
                self._history.append({'tgt': tgt, 'cost': m.cost})
                self.response_captured.emit(t_arr, y_arr, y0, tgt,
                                            {**m.as_dict(),
                                             'settle_band': self.config.settle_band})
                return m, t_arr, y_arr

            result = self.algorithm.run(
                evaluate=evaluate,
                log=lambda s: self.log.emit(s),
                is_cancelled=self.is_cancelled,
            )
            # restore center, then push best gains.
            self.backend.hold_position({joint: center})
            self.backend.set_gains(joint, result.gains)
            self.finished.emit(result)
        except StopIteration:
            self.log.emit('Tuning cancelled.')
            self.backend.hold_position({joint: center})
            self.finished.emit(None)
        except Exception as e:
            self.log.emit(f'Tuning error: {e}')
            self.finished.emit(None)


class Ros2PidTunerWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('ROS 2 PID Tuner')
        self.backend = RosBackend()
        self.backend.start()
        self._worker: TuneWorker | None = None
        self._worker_thread: threading.Thread | None = None
        self._joint_rows: dict[str, int] = {}
        # Live gain columns; replaced by the connected controller's field set.
        self._gain_fields: list[str] = list(DEFAULT_GAIN_FIELDS)

        self._build_ui()
        self._wire()

    # ----- UI ---------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Connection bar
        conn_box = QGroupBox('Controller')
        conn_layout = QHBoxLayout(conn_box)
        self.ed_namespace = QLineEdit('')
        self.ed_namespace.setPlaceholderText('namespace, e.g. /robot or empty')
        self.ed_namespace.setToolTip(
            'ROS namespace of the running ros2_control_node (empty if none).\n'
            'The tuner subscribes to <ns>/joint_states, <ns>/robot_description,\n'
            'and calls <ns>/<controller>/{get,set}_parameters.')
        self.ed_controller = QLineEdit('pid_controller')
        self.ed_controller.setToolTip(
            'Name of the controller instance to tune. The controller type is\n'
            'auto-detected: pid_controller (dof_names) or\n'
            'joint_trajectory_controller (joints).')
        self.btn_connect = QPushButton('Connect')
        self.btn_connect.setToolTip('Subscribe to joint states / URDF and read current gains.')
        self.btn_refresh = QPushButton('Read gains')
        self.btn_refresh.setToolTip('Re-read current gains and joint limits from the running controller.')
        self.btn_refresh.setEnabled(False)
        self.btn_urdf = QPushButton('Refill center from URDF')
        self.btn_urdf.setToolTip(
            'Fill each joint\'s center column with (lower+upper)/2 from the URDF\n'
            '`<limit>` tags. Edit the cell afterwards if you want a different\n'
            'reference position for the step test.')
        self.btn_urdf.setEnabled(False)
        conn_layout.addWidget(QLabel('namespace:'))
        conn_layout.addWidget(self.ed_namespace, 2)
        conn_layout.addWidget(QLabel('controller:'))
        conn_layout.addWidget(self.ed_controller, 1)
        conn_layout.addWidget(self.btn_connect)
        conn_layout.addWidget(self.btn_refresh)
        conn_layout.addWidget(self.btn_urdf)
        root.addWidget(conn_box)

        splitter = QSplitter(Qt.Horizontal)

        # Gain table (left)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.table = QTableWidget(0, 1 + len(self._gain_fields) + 1)
        self.table.setHorizontalHeaderLabels(
            ['joint', *self._gain_fields, CENTER_COL_HEADER])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setToolTip(
            'Per-joint gains. Edit cells then "Apply gains (live)".\n'
            'center column: reference position used by step test / autotune.\n'
            'Auto-filled from URDF; edit to override.')
        left_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton('Apply gains (live)')
        self.btn_apply.setToolTip(
            'Push the table\'s P/I/D/clamp values to the live controller via\n'
            'set_parameters. Effective immediately. The center column is NOT\n'
            'pushed to the controller; it is only used for step / autotune.')
        self.btn_load_yaml = QPushButton('Load YAML...')
        self.btn_load_yaml.setToolTip('Load gains from a pid_controller.yaml file.')
        self.btn_save_yaml = QPushButton('Save YAML...')
        self.btn_save_yaml.setToolTip('Save current table gains to a YAML file.')
        btn_row.addWidget(self.btn_apply)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_load_yaml)
        btn_row.addWidget(self.btn_save_yaml)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # Right pane: tuning + plot + log
        right = QWidget()
        right_layout = QVBoxLayout(right)

        tune_box = QGroupBox('Auto-tune')
        tune_form = QFormLayout(tune_box)
        self.cmb_joint = QComboBox()
        self.cmb_joint.setToolTip('Joint to tune. Step test moves only this joint;\n'
                                  'others are held at their last commanded position.')
        self.cmb_algo = QComboBox()
        for name in ALGORITHMS.keys():
            self.cmb_algo.addItem(name)
        # Default: Bayesian (last entry)
        self.cmb_algo.setCurrentIndex(self.cmb_algo.count() - 1)
        self.cmb_algo.setToolTip(
            'Algorithm:\n'
            ' - Ziegler-Nichols (open-loop, FOPDT): one step → identify K,τ,L → ZN PID rule.\n'
            ' - Cohen-Coon (FOPDT): same fit, more conservative (lower overshoot than ZN).\n'
            ' - Chien-Hrones-Reswick: FOPDT-based, default 0% overshoot setpoint mode.\n'
            ' - Internal Model Control (IMC): FOPDT-based, λ=closed-loop time constant\n'
            '   (max(0.1τ, 0.8L)). Smooth, low-overshoot.\n'
            ' - Relay Feedback (Astrom-Hagglund): probes increasing P → first oscillation\n'
            '   gives K_u, T_u → Tyreus-Luyben PID rule.\n'
            ' - Bayesian Optimization (GP+EI): black-box, samples 15-30 candidates,\n'
            '   directly minimises the cost (overshoot/settle/SSE weighted sum).\n'
            '   Recommended when the response is non-FOPDT (gear backlash, friction).')
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(-1.5, 1.5)
        self.spin_step.setSingleStep(0.05)
        self.spin_step.setDecimals(3)
        self.spin_step.setValue(0.20)
        self.spin_step.setToolTip(
            'Step displacement Δ from the joint\'s center [rad].\n'
            'Target = center + Δ (or center − Δ alternately, to cancel drift).\n'
            'Gets clipped into the safe range (URDF limits trimmed by safety margin).')
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.5, 10.0)
        self.spin_duration.setSingleStep(0.1)
        self.spin_duration.setValue(2.0)
        self.spin_duration.setToolTip(
            'Capture window per trial [s]. Total time the response is observed\n'
            'after each step. Increase for slow plants; decrease to make autotune faster.')
        self.spin_settle = QDoubleSpinBox()
        self.spin_settle.setRange(0.1, 5.0)
        self.spin_settle.setSingleStep(0.1)
        self.spin_settle.setValue(0.7)
        self.spin_settle.setToolTip(
            'Settle time at center before each step [s]. Larger = more reliable\n'
            'starting state, but each iteration is slower.')
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(2, 100)
        self.spin_iter.setValue(20)
        self.spin_iter.setToolTip(
            'Number of evaluations the autotuner performs.\n'
            ' - FOPDT methods (ZN/Cohen-Coon/CHR/IMC) need only 2.\n'
            ' - Relay Feedback uses 5–10.\n'
            ' - Bayesian Optimization needs 15–30 to converge.')
        self.spin_safety = QDoubleSpinBox()
        self.spin_safety.setRange(0.0, 0.45)
        self.spin_safety.setSingleStep(0.05)
        self.spin_safety.setValue(0.20)
        self.spin_safety.setToolTip(
            'Safety margin as a fraction of the URDF joint range, trimmed off\n'
            'BOTH ends. e.g. 0.20 means [lower+0.2*range, upper-0.2*range] is\n'
            'the allowed zone for steps. Anything outside contributes a heavy\n'
            'cost penalty so the autotuner learns to avoid it.')

        self.spin_w_over = QDoubleSpinBox(); self.spin_w_over.setRange(0.0, 100.0)
        self.spin_w_over.setValue(1.0); self.spin_w_over.setSingleStep(0.1)
        self.spin_w_over.setToolTip('Weight on overshoot term in cost. Increase to penalise overshoot.')
        self.spin_w_settle = QDoubleSpinBox(); self.spin_w_settle.setRange(0.0, 100.0)
        self.spin_w_settle.setValue(1.0); self.spin_w_settle.setSingleStep(0.1)
        self.spin_w_settle.setToolTip('Weight on settling time. Increase to prefer faster settling.')
        self.spin_w_sse = QDoubleSpinBox(); self.spin_w_sse.setRange(0.0, 100.0)
        self.spin_w_sse.setValue(5.0); self.spin_w_sse.setSingleStep(0.5)
        self.spin_w_sse.setToolTip('Weight on steady-state error. Increase for tighter accuracy.')

        # gain bounds for BO
        self.spin_p_max = QDoubleSpinBox(); self.spin_p_max.setRange(0.1, 100.0)
        self.spin_p_max.setValue(8.0); self.spin_p_max.setSingleStep(0.5)
        self.spin_p_max.setToolTip('Upper P bound used by Bayesian Optimization & Relay Feedback.\n'
                                   'Lower → safer but may underexplore. Default 8.0.')
        self.spin_i_max = QDoubleSpinBox(); self.spin_i_max.setRange(0.0, 50.0)
        self.spin_i_max.setValue(1.5); self.spin_i_max.setSingleStep(0.1)
        self.spin_i_max.setToolTip('Upper I bound for BO. Large I → integrator wind-up risk.')
        self.spin_d_max = QDoubleSpinBox(); self.spin_d_max.setRange(0.0, 10.0)
        self.spin_d_max.setValue(0.3); self.spin_d_max.setSingleStep(0.05)
        self.spin_d_max.setToolTip('Upper D bound for BO. Large D amplifies measurement noise.')

        tune_form.addRow('joint:', self.cmb_joint)
        tune_form.addRow('algorithm:', self.cmb_algo)
        tune_form.addRow('step Δ from center [rad]:', self.spin_step)
        tune_form.addRow('capture window [s]:', self.spin_duration)
        tune_form.addRow('settle at center [s]:', self.spin_settle)
        tune_form.addRow('iterations:', self.spin_iter)
        tune_form.addRow('safety margin (fraction of URDF range):', self.spin_safety)
        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel('w_overshoot'))
        weight_row.addWidget(self.spin_w_over)
        weight_row.addWidget(QLabel('w_settle'))
        weight_row.addWidget(self.spin_w_settle)
        weight_row.addWidget(QLabel('w_sse'))
        weight_row.addWidget(self.spin_w_sse)
        tune_form.addRow('cost weights:', weight_row)
        bounds_row = QHBoxLayout()
        bounds_row.addWidget(QLabel('P max'))
        bounds_row.addWidget(self.spin_p_max)
        bounds_row.addWidget(QLabel('I max'))
        bounds_row.addWidget(self.spin_i_max)
        bounds_row.addWidget(QLabel('D max'))
        bounds_row.addWidget(self.spin_d_max)
        tune_form.addRow('gain search bounds:', bounds_row)

        action_row = QHBoxLayout()
        self.btn_step = QPushButton('Run step test')
        self.btn_step.setToolTip(
            'One-shot: hold at center, settle, then step to (center + Δ).\n'
            'Uses the current PID gains. Plots the response with metrics.')
        self.btn_tune = QPushButton('Auto-tune selected joint')
        self.btn_tune.setToolTip(
            'Run the chosen algorithm on the selected joint. Best gains are\n'
            'auto-applied to the controller and the table at the end.')
        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip('Stop after the current trial; restore center.')
        action_row.addWidget(self.btn_step)
        action_row.addWidget(self.btn_tune)
        action_row.addWidget(self.btn_cancel)
        tune_form.addRow(action_row)
        right_layout.addWidget(tune_box)

        self.plot = StepResponsePlot()
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.plot, 2)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText('log...')
        right_layout.addWidget(self.log_view, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        self.status_label = QLabel('disconnected')
        self.status_label.setStyleSheet('color: gray;')
        root.addWidget(self.status_label)

    def _wire(self) -> None:
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_urdf.clicked.connect(self._on_urdf_refill)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_load_yaml.clicked.connect(self._on_load_yaml)
        self.btn_save_yaml.clicked.connect(self._on_save_yaml)
        self.btn_step.clicked.connect(self._on_step_test)
        self.btn_tune.clicked.connect(self._on_tune)
        self.btn_cancel.clicked.connect(self._on_cancel)

        self.backend.log.connect(self._append_log)
        self.backend.connection_changed.connect(self._on_conn_changed)

    # ----- handlers ---------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _on_conn_changed(self, ok: bool, msg: str) -> None:
        self.status_label.setText(msg)
        self.status_label.setStyleSheet('color: green;' if ok else 'color: red;')
        self.btn_refresh.setEnabled(ok)
        self.btn_urdf.setEnabled(ok)
        if ok:
            self._populate_joints()

    def _rebuild_table_columns(self) -> None:
        """Sync the gain columns to the connected controller's field set."""
        self._gain_fields = list(self.backend.gain_fields())
        self.table.setColumnCount(1 + len(self._gain_fields) + 1)
        self.table.setHorizontalHeaderLabels(
            ['joint', *self._gain_fields, CENTER_COL_HEADER])

    def _populate_joints(self) -> None:
        self._rebuild_table_columns()
        joints = self.backend.joints()
        self.cmb_joint.clear()
        self.cmb_joint.addItems(joints)
        self.table.setRowCount(len(joints))
        self._joint_rows = {}
        for r, j in enumerate(joints):
            self._joint_rows[j] = r
            self.table.setItem(r, 0, QTableWidgetItem(j))
            self.table.item(r, 0).setFlags(Qt.ItemIsEnabled)
            for c in range(1, self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem('0.0'))
        self._on_refresh()
        # try URDF refill (may already be available via latched topic)
        self._on_urdf_refill(silent=True)

    def _center_col(self) -> int:
        return 1 + len(self._gain_fields)

    def _on_urdf_refill(self, silent: bool = False) -> None:
        limits = self.backend.joint_limits()
        if not limits:
            if not silent:
                self._append_log('No URDF limits available yet.')
            return
        col = self._center_col()
        filled = 0
        for j, row in self._joint_rows.items():
            if j not in limits:
                continue
            lo, up = limits[j]
            self.table.setItem(row, col, QTableWidgetItem(f'{(lo + up) / 2.0:.4f}'))
            filled += 1
        self._append_log(f'URDF refill: set center for {filled}/{len(self._joint_rows)} joints.')

    def _on_connect(self) -> None:
        ns = self.ed_namespace.text().strip()
        ctrl = self.ed_controller.text().strip() or 'pid_controller'
        self.backend.connect_controller(ns, ctrl)

    def _on_refresh(self) -> None:
        gains = self.backend.get_all_gains()
        if not gains:
            self._append_log('Could not read current gains.')
            return
        for j, row in self._joint_rows.items():
            g = gains.get(j, {})
            for c, f in enumerate(self._gain_fields, start=1):
                self.table.setItem(row, c, QTableWidgetItem(f'{g.get(f, 0.0):.4f}'))

    def _row_to_gains(self, joint: str) -> dict[str, float]:
        row = self._joint_rows[joint]
        out = {}
        for c, f in enumerate(self._gain_fields, start=1):
            try:
                out[f] = float(self.table.item(row, c).text())
            except (ValueError, AttributeError):
                out[f] = 0.0
        return out

    def _row_center(self, joint: str) -> float:
        row = self._joint_rows[joint]
        try:
            return float(self.table.item(row, self._center_col()).text())
        except (ValueError, AttributeError):
            return 0.0

    def _set_row_center(self, joint: str, value: float) -> None:
        row = self._joint_rows[joint]
        self.table.setItem(row, self._center_col(), QTableWidgetItem(f'{value:.4f}'))

    def _safe_range(self, joint: str) -> tuple[float, float]:
        margin = self.spin_safety.value()
        lo, hi = self.backend.safe_range(joint, margin=margin,
                                         fallback=(-1e9, 1e9))
        # if fallback (no URDF), return wide range
        return lo, hi

    def _on_apply(self) -> None:
        for j in self._joint_rows:
            g = self._row_to_gains(j)
            ok = self.backend.set_gains(j, g)
            self._append_log(f'apply {j}: {"OK" if ok else "FAILED"}')

    def _on_load_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load PID gains YAML', str(Path.home()), 'YAML (*.yaml *.yml)')
        if not path:
            return
        try:
            joints, gains = load_gains(path)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load: {e}')
            return
        for j in joints:
            if j not in self._joint_rows:
                continue
            row = self._joint_rows[j]
            g = gains.get(j, {})
            for c, f in enumerate(self._gain_fields, start=1):
                self.table.setItem(row, c, QTableWidgetItem(f'{g.get(f, 0.0):.4f}'))
        self._append_log(f'Loaded {len(joints)} joints from {path}')

    def _on_save_yaml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save PID gains YAML', str(Path.home() / 'pid_controller.yaml'),
            'YAML (*.yaml *.yml)')
        if not path:
            return
        joints = list(self._joint_rows.keys())
        gains = {j: self._row_to_gains(j) for j in joints}
        try:
            if Path(path).exists():
                save_gains(path, joints, gains)
            else:
                self._write_new_yaml(path, joints, gains)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save: {e}')
            return
        self._append_log(f'Saved gains to {path}')

    def _write_new_yaml(self, path: str, joints: list[str], gains: dict) -> None:
        import yaml
        controller = self.ed_controller.text().strip() or 'pid_controller'
        doc = self.backend.adapter().new_yaml_doc(controller, joints, gains)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)

    def _make_config(self, joint: str) -> TunerConfig:
        return TunerConfig(
            joint=joint,
            target_delta=self.spin_step.value(),
            duration=self.spin_duration.value(),
            settle_time=self.spin_settle.value(),
            max_iter=self.spin_iter.value(),
            p_bounds=(0.05, self.spin_p_max.value()),
            i_bounds=(0.0, self.spin_i_max.value()),
            d_bounds=(0.0, self.spin_d_max.value()),
            weights=(self.spin_w_over.value(),
                     self.spin_w_settle.value(),
                     self.spin_w_sse.value()),
            safety_margin=self.spin_safety.value(),
        )

    def _on_step_test(self) -> None:
        joint = self.cmb_joint.currentText()
        if not joint:
            return
        try:
            center = self._row_center(joint)
            safe_lo, safe_hi = self._safe_range(joint)
            delta = self.spin_step.value()
            tgt = float(np.clip(center + delta, safe_lo, safe_hi))
            self.backend.hold_position({joint: center})
            time.sleep(self.spin_settle.value())
            t_arr, y_arr = self.backend.run_step(
                joint, tgt, duration=self.spin_duration.value())
            y0 = float(y_arr[0]) if y_arr.size else center
            m = compute_step_metrics(
                t_arr, y_arr, y0=y0, y_target=tgt,
                weights=(self.spin_w_over.value(),
                         self.spin_w_settle.value(),
                         self.spin_w_sse.value()),
                y_safe_min=safe_lo, y_safe_max=safe_hi)
            self.plot.show_response(t_arr, y_arr, target=tgt, initial=y0,
                                    metrics={**m.as_dict(), 'settle_band': 0.02})
            self._append_log(
                f'step {joint}: center={center:.3f} target={tgt:.3f} '
                f'safe=[{safe_lo:.3f},{safe_hi:.3f}]  {m.as_dict()}')
            # Return to center
            self.backend.hold_position({joint: center})
        except Exception as e:
            self._append_log(f'step error: {e}')

    def _on_tune(self) -> None:
        joint = self.cmb_joint.currentText()
        if not joint:
            return
        algo_name = self.cmb_algo.currentText()
        algo_cls = ALGORITHMS[algo_name]
        cfg = self._make_config(joint)
        center = self._row_center(joint)
        safe_lo, safe_hi = self._safe_range(joint)
        if abs(self.spin_step.value()) > (safe_hi - safe_lo) * 0.5:
            self._append_log(
                f'WARNING: step Δ ({self.spin_step.value():.3f}) '
                f'exceeds half of safe range [{safe_lo:.3f},{safe_hi:.3f}]; will be clipped.')

        worker = TuneWorker(self.backend, joint, algo_cls, cfg,
                            center=center, safe_lo=safe_lo, safe_hi=safe_hi)
        worker.log.connect(self._append_log)
        worker.response_captured.connect(self._on_response_captured)
        worker.finished.connect(self._on_tune_finished)
        self._worker = worker
        self._worker_thread = threading.Thread(target=worker.run, daemon=True,
                                               name=f'tune-{algo_name}')
        self._worker_thread.start()
        self.btn_tune.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._append_log(
            f'>>> Auto-tuning {joint} with {algo_name} '
            f'around center={center:.3f}, Δ={self.spin_step.value():.3f}, '
            f'safe=[{safe_lo:.3f},{safe_hi:.3f}]')

    def _on_response_captured(self, t, y, initial, target, metrics_dict) -> None:
        self.plot.show_response(t, y, target=target, initial=initial,
                                metrics=metrics_dict)

    def _on_tune_finished(self, result) -> None:
        self.btn_tune.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if result is None:
            self._append_log('Tuning ended without result.')
            return
        self._append_log(f'>>> Best gains: {result.gains}  cost={result.cost:.3f}  notes={result.notes}')
        joint = self.cmb_joint.currentText()
        if joint in self._joint_rows:
            row = self._joint_rows[joint]
            for c, f in enumerate(self._gain_fields, start=1):
                self.table.setItem(row, c, QTableWidgetItem(f'{result.gains.get(f, 0.0):.4f}'))

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._append_log('Cancellation requested.')

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        try:
            self.backend.stop()
        finally:
            super().closeEvent(event)
