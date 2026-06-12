from setuptools import setup
from glob import glob

package_name = 'dg_pid_tuner_gui'

setup(
    name=package_name,
    version='0.1.0',
    packages=[
        package_name,
        f'{package_name}.algorithms',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
         [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tesollo',
    maintainer_email='hchsk25@gmail.com',
    description='PyQt5 GUI for tuning ros2_control PidController gains on Tesollo grippers.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'dg_pid_tuner_gui = {package_name}.main:main',
        ],
    },
)
