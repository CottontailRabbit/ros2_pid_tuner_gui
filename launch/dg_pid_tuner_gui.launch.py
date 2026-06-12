"""Launch the standalone PyQt5 PID tuner GUI."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='/dg3f_m',
        description='Controller namespace, e.g. /dg3f_m')
    controller_arg = DeclareLaunchArgument(
        'controller', default_value='pid_controller',
        description='Controller name')

    gui_node = Node(
        package='dg_pid_tuner_gui',
        executable='dg_pid_tuner_gui',
        name='dg_pid_tuner_gui',
        output='screen',
        parameters=[{
            'namespace': LaunchConfiguration('namespace'),
            'controller': LaunchConfiguration('controller'),
        }],
    )

    return LaunchDescription([namespace_arg, controller_arg, gui_node])
