"""Launch the standalone PyQt5 PID tuner GUI."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='',
        description='Controller namespace, e.g. /robot (empty if none)')
    controller_arg = DeclareLaunchArgument(
        'controller', default_value='pid_controller',
        description='Controller name (pid_controller or joint_trajectory_controller)')

    gui_node = Node(
        package='ros2_pid_tuner_gui',
        executable='ros2_pid_tuner_gui',
        name='ros2_pid_tuner_gui',
        output='screen',
        parameters=[{
            'namespace': LaunchConfiguration('namespace'),
            'controller': LaunchConfiguration('controller'),
        }],
    )

    return LaunchDescription([namespace_arg, controller_arg, gui_node])
