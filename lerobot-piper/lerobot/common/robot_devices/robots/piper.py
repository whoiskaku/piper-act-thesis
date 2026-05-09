"""
    Teleoperation Agilex Piper with a PS5 controller    
"""

import sys
import time
import torch
import numpy as np
from dataclasses import dataclass, field, replace
from pathlib import Path

from lerobot.common.robot_devices.teleop.gamepad import SixAxisArmController
from lerobot.common.robot_devices.motors.utils import get_motor_names, make_motors_buses_from_configs
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.robots.configs import PiperRobotConfig

_TSND_READER_DIR = Path(__file__).resolve().parents[4] / "scripts" / "tsnd_real"
if str(_TSND_READER_DIR) not in sys.path:
    sys.path.insert(0, str(_TSND_READER_DIR))

IMU_STALE_THRESHOLD_S = 0.1  # get_latest 超过这个龄视为掉线

class PiperRobot:
    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        if config is None:
            config = PiperRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.inference_time = self.config.inference_time # if it is inference time
        
        # build cameras
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        # build piper motors
        self.piper_motors = make_motors_buses_from_configs(self.config.follower_arm)
        self.arm = self.piper_motors['main']
        
        # build gamepad teleop
        if not self.inference_time:
            self.teleop = SixAxisArmController()
        else:
            self.teleop = None

        # build TSND151 IMU reader (not started until connect())
        self.imu_reader = None
        if getattr(self.config, "enable_imu", False):
            from tsnd151_reader import TSND151Reader
            self.imu_reader = TSND151Reader(
                device_path=self.config.tsnd_device,
                interval_ms=self.config.tsnd_interval_ms,
            )

        self.logs = {}
        self.is_connected = False

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,   
            }
        return cam_ft

    
    @property
    def motor_features(self) -> dict:
        action_names = get_motor_names(self.piper_motors)
        state_names = get_motor_names(self.piper_motors)
        features = {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }
        if self.imu_reader is not None:
            features["observation.imu_real"] = {
                "dtype": "float32",
                "shape": (6,),
                "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            }
        return features

    def _read_imu_tensor(self) -> torch.Tensor:
        """Read latest IMU sample, assert freshness. Raises if reader is stale or missing."""
        latest = self.imu_reader.get_latest()
        if latest is None:
            raise RuntimeError("IMU reader has no sample (reader died or never started)")
        host_t, _, acc, gyro = latest
        age = time.monotonic() - host_t
        if age > IMU_STALE_THRESHOLD_S:
            raise RuntimeError(
                f"IMU sample stale: {age*1000:.1f}ms old (>{IMU_STALE_THRESHOLD_S*1000:.0f}ms). "
                f"TSND151 likely disconnected."
            )
        imu6 = np.concatenate([acc, gyro]).astype(np.float32)
        return torch.from_numpy(imu6)
    
    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)


    def connect(self) -> None:
        """Connect piper and cameras"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )
        
        # connect piper
        self.arm.connect(enable=True)
        print("piper conneted")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected
            print(f"camera {name} conneted")
        
        # start IMU reader + block until first sample arrives (hard contract)
        if self.imu_reader is not None:
            self.imu_reader.start()
            deadline = time.monotonic() + 1.0
            while self.imu_reader.get_latest() is None:
                if time.monotonic() > deadline:
                    self.imu_reader.stop()
                    self.imu_reader = None
                    raise RuntimeError(
                        f"TSND151 at {self.config.tsnd_device} produced no sample within 1s. "
                        f"Check device path, USB connection, and permissions."
                    )
                time.sleep(0.02)
            print("IMU (TSND151) connected")

        print("All connected")
        self.is_connected = True

        self.run_calibration()


    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""
        # move piper to home position, disable
        if not self.inference_time:
            self.teleop.stop()

        # disconnect piper
        self.arm.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.arm.connect(enable=False)

        # disconnect cameras
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        # stop IMU reader
        if self.imu_reader is not None:
            self.imu_reader.stop()

        self.is_connected = False


    def run_calibration(self):
        """move piper to the home position"""
        if not self.is_connected:
            raise ConnectionError()
        
        self.arm.apply_calibration()
        if not self.inference_time:
            self.teleop.reset()



    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()
        
        if self.teleop is None and self.inference_time:
            self.teleop = SixAxisArmController()

        # read target pose state as 
        before_read_t = time.perf_counter()
        state = self.arm.read() # read current joint position from robot
        action = self.teleop.get_action() # target joint position from gamepad
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # do action
        before_write_t = time.perf_counter()
        target_joints = list(action.values())
        self.arm.write(target_joints)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if not record_data:
            return
        
        state = torch.as_tensor(list(state.values()), dtype=torch.float32)
        action = torch.as_tensor(list(action.values()), dtype=torch.float32)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        if self.imu_reader is not None:
            obs_dict["observation.imu_real"] = self._read_imu_tensor()

        return obs_dict, action_dict



    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Write the predicted actions from policy to the motors"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        # send to motors, torch to list
        target_joints = action.tolist()
        self.arm.write(target_joints)

        return action



    def capture_observation(self) -> dict:
        """capture current images and joint positions"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )
        
        # read current joint positions
        before_read_t = time.perf_counter()
        state = self.arm.read()  # 6 joints + 1 gripper
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        state = torch.as_tensor(list(state.values()), dtype=torch.float32)

        # read images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        if self.imu_reader is not None:
            obs_dict["observation.imu_real"] = self._read_imu_tensor()

        return obs_dict
    
    def teleop_safety_stop(self):
        """ move to home position after record one episode """
        self.run_calibration()

    
    def __del__(self):
        if self.is_connected:
            self.disconnect()
            if not self.inference_time:
                self.teleop.stop()
                
