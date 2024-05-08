from setuptools import setup, find_packages

setup(
    name="robot_infra",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gym",
        "pyrealsense2",
        "pymodbus==2.5.3",
        "opencv-python",
        "pyquaternion",
        "pyspacemouse",
        "hidapi",
        "pyyaml",
        "rospkg",
        "scipy",
        "requests",
        "flask",
        "defusedxml",
        "absl-py==1.4.0",
        "pynput"
    ],
)
