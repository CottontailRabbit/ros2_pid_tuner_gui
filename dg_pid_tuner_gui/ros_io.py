"""rclpy backend that drives a remote pid_controller.

Runs an rclpy executor in a background thread so the PyQt GUI stays
responsive. All public methods are safe to call from the GUI thread;
results are returned synchronously when feasible, or via Qt signals
for streaming (joint_states) and finalised (step done) events.
"""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
                       QoSDurabilityPolicy)

from rcl_interfaces.srv import GetParameters, SetParameters, ListParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from sensor_msgs.msg import JointState
from std_msgs.msg import String

try:
    from control_msgs.msg import MultiDOFCommand
    HAVE_MULTIDOF = True
except ImportError:
    HAVE_MULTIDOF = False

from PyQt5.QtCore import QObject, pyqtSignal


GAIN_FIELDS = ('p', 'i', 'd', 'i_clamp_max', 'i_clamp_min')


@dataclass
class StepCapture:
    target: float = 0.0
    initial: float = 0.0
    joint: str = ''
    t0: float = 0.0
    duration: float = 2.0
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    active: bool = False
    done_event: Optional[threading.Event] = None


class RosBackend(QObject):
    """Bridges rclpy and Qt signals."""

    log = pyqtSignal(str)
    connection_changed = pyqtSignal(bool, str)
    joint_state_sample = pyqtSignal(str, float, float)  # joint, t, pos
    step_finished = pyqtSignal(str, object, object, float)  # joint, t[], y[], target

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._node: Optional[Node] = None
        self._joint_sub = None
        self._ref_pub = None

        self._namespace: str = ''
        self._controller: str = 'pid_controller'
        self._joints: list[str] = []
        self._latest_pos: dict[str, float] = {}
        self._capture: Optional[StepCapture] = None
        self._urdf_sub = None
        self._joint_limits: dict[str, tuple[float, float]] = {}

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._node is not None:
                return
            if not rclpy.ok():
                rclpy.init()
            self._node = Node('dg_pid_tuner_gui')
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, daemon=True, name='dg_pid_tuner_spin'
            )
            self._spin_thread.start()
            self._emit_log('rclpy backend started.')

    def stop(self) -> None:
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
            self._executor = None
            self._node = None
            self._spin_thread = None
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    # -- connection ----------------------------------------------------

    def connect_controller(self, namespace: str, controller: str = 'pid_controller') -> bool:
        """Subscribe to joint_states and resolve the controller's joints."""
        if self._node is None:
            self.start()
        with self._lock:
            self._namespace = namespace.rstrip('/')
            self._controller = controller
            ns = self._namespace if self._namespace.startswith('/') else ('/' + self._namespace if self._namespace else '')

            joints = self._fetch_dof_names(ns, controller)
            if not joints:
                self.connection_changed.emit(False, f'No dof_names from {ns}/{controller}')
                return False
            self._joints = joints

            # joint_states subscription
            qos = QoSProfile(depth=50, reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST)
            js_topic = f'{ns}/joint_states'
            if self._joint_sub is not None:
                self._node.destroy_subscription(self._joint_sub)
            self._joint_sub = self._node.create_subscription(
                JointState, js_topic, self._on_joint_state, qos)

            # reference publisher
            if HAVE_MULTIDOF:
                ref_topic = f'{ns}/{controller}/reference'
                if self._ref_pub is not None:
                    self._node.destroy_publisher(self._ref_pub)
                self._ref_pub = self._node.create_publisher(MultiDOFCommand, ref_topic, 10)
            else:
                self._emit_log('control_msgs/MultiDOFCommand not available; reference publishing disabled.')

            # URDF subscription (latched/transient_local) for joint limits
            qos_latched = QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
            )
            urdf_topic = f'{ns}/robot_description'
            if self._urdf_sub is not None:
                self._node.destroy_subscription(self._urdf_sub)
            self._joint_limits = {}
            self._urdf_sub = self._node.create_subscription(
                String, urdf_topic, self._on_urdf, qos_latched)

            self.connection_changed.emit(True, f'Connected to {ns}/{controller}, {len(joints)} joints.')
            return True

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return dict(self._joint_limits)

    def joint_center(self, joint: str, fallback: float = 0.0) -> float:
        lim = self._joint_limits.get(joint)
        if lim is None:
            return fallback
        return float((lim[0] + lim[1]) / 2.0)

    def safe_range(
        self, joint: str, margin: float = 0.2,
        fallback: tuple[float, float] = (-1.0, 1.0),
    ) -> tuple[float, float]:
        """Return [lower + margin*range, upper - margin*range]."""
        lim = self._joint_limits.get(joint)
        if lim is None:
            return fallback
        lo, up = float(lim[0]), float(lim[1])
        rng = up - lo
        if rng <= 0:
            return fallback
        m = max(0.0, min(0.45, margin))
        return (lo + m * rng, up - m * rng)

    def _on_urdf(self, msg: String) -> None:
        try:
            root = ET.fromstring(msg.data)
        except ET.ParseError:
            return
        limits: dict[str, tuple[float, float]] = {}
        for j in root.findall('joint'):
            jname = j.get('name')
            jtype = (j.get('type') or '').lower()
            if jtype in ('fixed', 'floating', ''):
                continue
            lim = j.find('limit')
            if lim is None:
                continue
            try:
                lo = float(lim.get('lower', 'nan'))
                up = float(lim.get('upper', 'nan'))
            except (TypeError, ValueError):
                continue
            if lo != lo or up != up or up <= lo:
                continue
            limits[jname] = (lo, up)
        if limits:
            self._joint_limits = limits
            self._emit_log(f'Parsed {len(limits)} joint limits from URDF.')

    def joints(self) -> list[str]:
        return list(self._joints)

    # -- parameter get/set --------------------------------------------

    def _param_full_name(self, joint: str, field_name: str) -> str:
        return f'gains.{joint}.{field_name}'

    def _service_name(self, suffix: str) -> str:
        ns = self._namespace if self._namespace.startswith('/') else ('/' + self._namespace if self._namespace else '')
        return f'{ns}/{self._controller}/{suffix}'

    def _fetch_dof_names(self, ns: str, controller: str) -> list[str]:
        client = self._node.create_client(GetParameters, f'{ns}/{controller}/get_parameters')
        if not client.wait_for_service(timeout_sec=3.0):
            self._emit_log(f'Service {ns}/{controller}/get_parameters not available.')
            return []
        req = GetParameters.Request()
        req.names = ['dof_names']
        future = client.call_async(req)
        if not self._wait_future(future, 3.0):
            return []
        resp = future.result()
        if not resp or not resp.values:
            return []
        v = resp.values[0]
        if v.type == ParameterType.PARAMETER_STRING_ARRAY:
            return list(v.string_array_value)
        return []

    def get_all_gains(self) -> dict[str, dict[str, float]]:
        if not self._joints:
            return {}
        client = self._node.create_client(GetParameters, self._service_name('get_parameters'))
        if not client.wait_for_service(timeout_sec=3.0):
            self._emit_log('get_parameters service not available.')
            return {}
        names: list[str] = []
        for j in self._joints:
            for f in GAIN_FIELDS:
                names.append(self._param_full_name(j, f))
        req = GetParameters.Request()
        req.names = names
        future = client.call_async(req)
        if not self._wait_future(future, 3.0):
            return {}
        resp = future.result()
        if not resp or len(resp.values) != len(names):
            self._emit_log('get_parameters returned unexpected size.')
            return {}
        result: dict[str, dict[str, float]] = {j: {} for j in self._joints}
        for n, v in zip(names, resp.values):
            joint, field_name = self._parse_param(n)
            result.setdefault(joint, {})[field_name] = self._param_value_as_float(v)
        return result

    def set_gains(self, joint: str, gains: dict[str, float]) -> bool:
        client = self._node.create_client(SetParameters, self._service_name('set_parameters'))
        if not client.wait_for_service(timeout_sec=3.0):
            self._emit_log('set_parameters service not available.')
            return False
        params = []
        for f in GAIN_FIELDS:
            if f not in gains:
                continue
            p = Parameter()
            p.name = self._param_full_name(joint, f)
            p.value.type = ParameterType.PARAMETER_DOUBLE
            p.value.double_value = float(gains[f])
            params.append(p)
        req = SetParameters.Request()
        req.parameters = params
        future = client.call_async(req)
        if not self._wait_future(future, 3.0):
            return False
        resp = future.result()
        ok = bool(resp) and all(r.successful for r in resp.results)
        if not ok and resp is not None:
            for r in resp.results:
                if not r.successful:
                    self._emit_log(f'set_parameters failed: {r.reason}')
        return ok

    @staticmethod
    def _parse_param(name: str) -> tuple[str, str]:
        # gains.<joint>.<field>
        parts = name.split('.', 2)
        if len(parts) == 3:
            return parts[1], parts[2]
        return '', ''

    @staticmethod
    def _param_value_as_float(v: ParameterValue) -> float:
        if v.type == ParameterType.PARAMETER_DOUBLE:
            return float(v.double_value)
        if v.type == ParameterType.PARAMETER_INTEGER:
            return float(v.integer_value)
        return 0.0

    # -- step test -----------------------------------------------------

    def latest_position(self, joint: str) -> Optional[float]:
        return self._latest_pos.get(joint)

    def run_step(self, joint: str, target: float, duration: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """Blocking call: publish step reference and capture response."""
        if self._ref_pub is None:
            raise RuntimeError('No reference publisher available (control_msgs missing or not connected).')
        if joint not in self._joints:
            raise ValueError(f'Joint {joint!r} not in controller dof_names.')
        initial = self._latest_pos.get(joint)
        if initial is None:
            # wait briefly for a sample
            for _ in range(50):
                time.sleep(0.02)
                initial = self._latest_pos.get(joint)
                if initial is not None:
                    break
            if initial is None:
                raise RuntimeError(f'No joint_state for {joint} received yet.')

        cap = StepCapture(
            target=float(target), initial=float(initial), joint=joint,
            t0=time.monotonic(), duration=float(duration),
            done_event=threading.Event(), active=True,
        )
        with self._lock:
            self._capture = cap

        # publish step reference (target for selected joint, hold others at current)
        msg = MultiDOFCommand()
        msg.dof_names = list(self._joints)
        msg.values = [self._latest_pos.get(j, 0.0) for j in self._joints]
        try:
            idx = self._joints.index(joint)
            msg.values[idx] = float(target)
        except ValueError:
            pass
        # values_dot: leave empty (zero feedforward)
        msg.values_dot = []
        self._ref_pub.publish(msg)

        # wait for capture to finish
        cap.done_event.wait(timeout=duration + 1.0)
        with self._lock:
            self._capture = None

        t_arr = np.asarray(cap.times, dtype=float)
        y_arr = np.asarray(cap.values, dtype=float)
        self.step_finished.emit(joint, t_arr, y_arr, cap.target)
        return t_arr, y_arr

    def hold_position(self, positions: dict[str, float]) -> None:
        """Send a reference message holding given joint positions (others = latest)."""
        if self._ref_pub is None:
            return
        msg = MultiDOFCommand()
        msg.dof_names = list(self._joints)
        msg.values = [positions.get(j, self._latest_pos.get(j, 0.0)) for j in self._joints]
        msg.values_dot = []
        self._ref_pub.publish(msg)

    # -- callbacks -----------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        now = time.monotonic()
        for name, pos in zip(msg.name, msg.position):
            self._latest_pos[name] = float(pos)
            self.joint_state_sample.emit(name, now, float(pos))
        cap = self._capture
        if cap is None or not cap.active:
            return
        try:
            idx = msg.name.index(cap.joint)
        except ValueError:
            return
        t = now - cap.t0
        cap.times.append(t)
        cap.values.append(float(msg.position[idx]))
        if t >= cap.duration and cap.done_event is not None:
            cap.active = False
            cap.done_event.set()

    # -- helpers -------------------------------------------------------

    def _wait_future(self, future, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            time.sleep(0.01)
        return future.done()

    def _emit_log(self, msg: str) -> None:
        self.log.emit(msg)
