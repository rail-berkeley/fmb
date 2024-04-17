# Robot Infra
![](../docs/robot_infra_interfaces.jpg)

The robot infra used to collect the dataset is released as part of the [serl](https://github.com/rail-berkeley/serl/tree/main/serl_robot_infra) project and works for both Franka Emikda Panda and the newer Franka Research 3 arms. 

All robot code is structured as follows:
There is a Flask server which sends commands to the robot via ROS. There is a gym env for the robot which communicates with the Flask server via post requests.

- `robot_server`: hosts a Flask server which sends commands to the robot via ROS
- `franka_env`: gym env for the robot which communicates with the Flask server via post requests


### Prerequisite
- ROS Noetic
- Franka Panda or Franka Research 3 arm and gripper
- `libfranka>=0.8.0` and `franka_ros>=0.8.0` installed according to [Franka FCI Documentation](https://frankaemika.github.io/docs/installation_linux.html)

### Install
```bash
cd robot_infra
conda create -n fmb_robot_infra python=3.9
conda activat fmb_robot_infra
pip install -e .
```
Install the [`serl_franka_controllers`](https://github.com/rail-berkeley/serl_franka_controllers) package.

    sudo apt-get install ros-serl_franka_controllers

### Usage
1. Launch the robot server, which run the robot controller and a Flask server which streams robot commands to the gym envionrment using HTTP requests. 
    ```bash
    conda activat fmb_robot_infra
    python franka_server.py --robot_ip=<robot_IP>
2. Create an instance of the gym environment in a second terminal.
    ```python
    import gym
    import fmb_robot_infra
    env = gym.make("Franka-FMB-v0")
    ```
This should start ROS node impedence controller and the HTTP server. You can test that things are running by trying to move the end effector around, if the impedence controller is running it should be compliant.

The HTTP server is used to communicate between the ROS controller and gym environments. Possible HTTP requests include:

| Request | Description |
| --- | --- |
| startimp | Stop the impedance controller |
| stopimp | Start the impedance controller |
| pose | Command robot to go to desired end-effector pose given in base frame (xyz+quaternion) |
| getpos | Return current end-effector pose in robot base frame (xyz+rpy)|
| getvel | Return current end-effector velocity in robot base frame |
| getforce | Return estimated force on end-effector |
| gettorque | Return estimated torque on end-effector |
| getq | Return current joint position |
| getdq | Return current joint velocity |
| getjacobian | Return current zero-jacobian |
| getstate | Return all robot states |
| jointreset | Perform joint reset |
| reset_gripper | Reset the gripper (Robotiq only) |
| get_gripper | Return current gripper position |
| close_gripper | Close the gripper completely |
| open_gripper | Open the gripper completely |
| clearerr | Clear errors |
| precision_mode | Update the impedance controller parameters to precision mode for resets|
| compliance_mode | Update the impedance controller parameters to compliance mode for task execution |

These commands can also be called in terminal. Useful ones include:
```bash
curl -X POST http://127.0.0.1:5000/activate_gripper # Activate gripper
curl -X POST http://127.0.0.1:5000/close_gripper # Close gripper
curl -X POST http://127.0.0.1:5000/open_gripper # Open gripper
curl -X POST http://127.0.0.1:5000/getpos # Print current end-effector pose
curl -X POST http://127.0.0.1:5000/jointreset # Perform joint reset
curl -X POST http://127.0.0.1:5000/precision_mode # Change the impedance controller to precision mode
curl -X POST http://127.0.0.1:5000/compliance_mode # Change the impedance controller to compliance mode
curl -X POST http://127.0.0.1:5000/stopimp # Stop the impedance controller
curl -X POST http://127.0.0.1:5000/startimp # Start the impedance controller (**Only run this after stopimp**)
```
