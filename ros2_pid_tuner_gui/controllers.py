"""Controller-type adapters.

Abstracts the differences between ros2_control controllers that expose
runtime-tunable PID gains, so the GUI/backend can drive any of them with
one code path. Currently supported:

  * ``pid_controller``                  (control_toolbox / PidController)
  * ``joint_trajectory_controller``     (JTC, topic interface)

Each adapter knows:
  * which parameter holds the DOF list (``dof_names`` vs ``joints``),
  * which per-joint gain fields the controller accepts,
  * how to create a command publisher and build a step/hold message,
  * a minimal YAML skeleton for "Save as new file".
"""
from __future__ import annotations

from typing import Iterable

from rclpy.duration import Duration

try:
    from control_msgs.msg import MultiDOFCommand
    HAVE_MULTIDOF = True
except ImportError:
    HAVE_MULTIDOF = False

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Default gain columns shown before a controller is connected.
DEFAULT_GAIN_FIELDS = ('p', 'i', 'd', 'i_clamp_max', 'i_clamp_min')


class ControllerAdapter:
    """Base adapter. Subclasses describe one controller type."""

    kind: str = 'base'
    #: parameter that holds the list of controlled DOFs / joints
    dof_param: str = 'dof_names'
    #: per-joint gain fields the controller exposes as ``gains.<joint>.<field>``
    gain_fields: tuple[str, ...] = DEFAULT_GAIN_FIELDS
    #: controller plugin type string, used when writing a fresh YAML
    plugin_type: str = ''

    def make_publisher(self, node, ns: str, controller: str):
        """Create the publisher used to command references. May return None."""
        raise NotImplementedError

    def build_command(self, joints: Iterable[str], values: Iterable[float],
                      time_from_start: float = 0.0):
        """Build the command message that drives ``joints`` to ``values``."""
        raise NotImplementedError

    def new_yaml_doc(self, controller: str, joints: list[str],
                     gains: dict[str, dict[str, float]]) -> dict:
        """Return a minimal parameter YAML document for this controller."""
        return {
            '/**/controller_manager': {
                'ros__parameters': {
                    'update_rate': 100,
                    controller: {'type': self.plugin_type},
                }
            },
            f'/**/{controller}': {
                'ros__parameters': {
                    self.dof_param: joints,
                    'gains': {j: gains.get(j, {}) for j in joints},
                }
            },
        }


class PidControllerAdapter(ControllerAdapter):
    kind = 'pid_controller'
    dof_param = 'dof_names'
    gain_fields = ('p', 'i', 'd', 'i_clamp_max', 'i_clamp_min')
    plugin_type = 'pid_controller/PidController'

    def make_publisher(self, node, ns: str, controller: str):
        if not HAVE_MULTIDOF:
            return None
        topic = f'{ns}/{controller}/reference'
        return node.create_publisher(MultiDOFCommand, topic, 10)

    def build_command(self, joints, values, time_from_start: float = 0.0):
        msg = MultiDOFCommand()
        msg.dof_names = list(joints)
        msg.values = [float(v) for v in values]
        msg.values_dot = []
        return msg

    def new_yaml_doc(self, controller, joints, gains):
        doc = super().new_yaml_doc(controller, joints, gains)
        params = doc[f'/**/{controller}']['ros__parameters']
        params['command_interface'] = 'effort'
        params['reference_and_state_interfaces'] = ['position']
        # keep gains last for readability
        params['gains'] = {j: gains.get(j, {}) for j in joints}
        return doc


class JointTrajectoryControllerAdapter(ControllerAdapter):
    kind = 'joint_trajectory_controller'
    dof_param = 'joints'
    gain_fields = ('p', 'i', 'd', 'i_clamp', 'ff_velocity_scale')
    plugin_type = 'joint_trajectory_controller/JointTrajectoryController'

    def make_publisher(self, node, ns: str, controller: str):
        topic = f'{ns}/{controller}/joint_trajectory'
        return node.create_publisher(JointTrajectory, topic, 10)

    def build_command(self, joints, values, time_from_start: float = 0.0):
        msg = JointTrajectory()
        msg.joint_names = list(joints)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in values]
        # A near-instant single waypoint makes the reference behave as a step.
        pt.time_from_start = Duration(seconds=max(0.0, float(time_from_start))).to_msg()
        msg.points = [pt]
        return msg

    def new_yaml_doc(self, controller, joints, gains):
        doc = super().new_yaml_doc(controller, joints, gains)
        params = doc[f'/**/{controller}']['ros__parameters']
        params['command_interfaces'] = ['effort']
        params['state_interfaces'] = ['position', 'velocity']
        params['gains'] = {j: gains.get(j, {}) for j in joints}
        return doc


# Detection order: each adapter is identified by its unique dof parameter.
ADAPTERS: tuple[type[ControllerAdapter], ...] = (
    PidControllerAdapter,
    JointTrajectoryControllerAdapter,
)
