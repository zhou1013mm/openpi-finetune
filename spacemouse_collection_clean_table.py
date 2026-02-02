"""
Teleoperating robot arm with a SpaceMouse to collect demonstration data
fps = 5

python task/spacemouse_collection_clean_table.py --task_name test --task_id 0

"""
import sys
sys.path.append("/home/czhpc/deoxys_codebase/deoxys_control/deoxys/")

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path
from easydict import EasyDict

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2
import imageio.v3 as iio

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import joint_interpolation_traj, follow_joint_traj

from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys_vision.camera.rs_interface import RSInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys_vision.threading.threading_utils import Worker

logger = get_deoxys_example_logger()


def get_rs_intrinsics_param(K_matrix: np.ndarray):
    """
    Args:
       K_matrix (np.ndarray): Numpy matrix of camera intrinsics

    Return:
       intrinsics_params (dict): a dictionary of intrinsics parameters, namely fx, fy, cx, cy
    """
    return {"fx": K_matrix[0][0], "fy": K_matrix[1][1], "cx": K_matrix[0][2], "cy": K_matrix[1][2]}


class RSCameraWorker(Worker):
    def __init__(
        self,
        camera_config: EasyDict = {},
        device_id=None,
        thread_safe: bool = True,
    ):

        # try:
        self.pipeline = rs.pipeline()

        self.config = rs.config()

        # Bind this pipeline to a specific RealSense device (serial number).
        # Without this, creating multiple pipelines will fight over the default device
        # and can trigger: xioctl(VIDIOC_S_FMT) failed, errno=16 (Device or resource busy)
        if device_id is not None:
            self.config.enable_device(str(device_id))

        # rs.config.enable_device_from_file(config, args.input)
        # Configure the pipeline to stream the depth stream

        self.enable_color = camera_config.enable_color
        self.enable_depth = camera_config.enable_depth
        if camera_config.enable_color:
            self.config.enable_stream(
                rs.stream.color,
                camera_config.color_cfg.img_w,
                camera_config.color_cfg.img_h,
                camera_config.color_cfg.img_format,
                camera_config.color_cfg.fps,
            )

        if camera_config.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                camera_config.depth_cfg.img_w,
                camera_config.depth_cfg.img_h,
                camera_config.depth_cfg.img_format,
                camera_config.depth_cfg.fps,
            )

        # Start streaming from file
        profile = self.pipeline.start(self.config)
        sensor_dep = profile.get_device().first_depth_sensor()
        sensor_dep.set_option(rs.option.enable_auto_exposure, 1)

        # # Create colorizer object (for depth)
        # colorizer = rs.colorizer()

        self.last_obs = None
        self.camera_config = camera_config

        self.calibration = {
            "color": {"intrinsics": None, "distortion": None},
            "depth": {"intrinsics": None, "distortion": None},
        }

        super().__init__()

    def get_intrinsics(self, key, mode=None):
        assert key in ["color", "depth"]
        if mode == "dict":
            return get_rs_intrinsics_param(self.calibration[key]["intrinsics"])
        else:
            return self.calibration[key]["intrinsics"]

    def get_distortion(self, key):
        assert key in ["color"]
        return self.calibration[key]["distortion"]

    def run(self) -> None:
        self.last_obs = EasyDict()

        self.profile = self.pipeline.get_active_profile()
        self.color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        # self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        color_intrinsics = self.color_profile.intrinsics
        # depth_intrinsics = self.depth_profile.intrinsics
        # print(depth_intrinsics)

        color_K_matrix = np.array(
            [
                [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ]
        )
        # depth_K_matrix = np.array(
        #     [
        #         [depth_intrinsics.fx, 0.0, depth_intrinsics.ppx],
        #         [0.0, depth_intrinsics.fy, depth_intrinsics.ppy],
        #         [0.0, 0.0, 1.0],
        #     ]
        # )
        depth_K_matrix = color_K_matrix

        self.calibration["color"]["intrinsics"] = color_K_matrix
        self.calibration["depth"]["intrinsics"] = depth_K_matrix
        # print(color_K_matrix)
        # print(depth_K_matrix)

        self.calibration["color"]["distortion"] = np.array(color_intrinsics.coeffs)
        self.calibration["depth"]["distortion"] = np.array(color_intrinsics.coeffs) # np.array(depth_intrinsics.coeffs)

        # color_intrinsics = self.color_profile.get_intrinsics()
        # return {'fx': color_intrinsics.fx,
        #         'fy': color_intrinsics.fy,
        #         'cx': color_intrinsics.ppx,
        #         'cy': color_intrinsics.ppy}, color_intrinsics.width, color_intrinsics.height

        align = rs.align(rs.stream.color)
        while not self._halt:
            frames = self.pipeline.wait_for_frames()
            if frames is None:
                continue
            if self.enable_color:
                self.last_obs["color"] = np.asanyarray(frames.get_color_frame().get_data())
            if self.enable_depth:
                self.last_obs["unaligned_depth"] = np.asanyarray(
                    frames.get_depth_frame().get_data()
                )
                frames = align.process(frames)
                self.last_obs["depth"] = np.asanyarray(frames.get_depth_frame().get_data())
        self.pipeline.stop()
        del self.pipeline

    def save_img(self, img_name):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # cv2.imwrite(img_name, color_image)
        iio.imwrite(img_name, color_image)


class RSInterface:
    """ "
    This is the Python Interface for getting images from Realsense D435i.
    """

    def __init__(
        self,
        device_id,
        color_cfg: dict = None,
        depth_cfg: dict = None,
        pc_cfg: dict = None,
    ):

        if color_cfg is not None:
            self.color_cfg = color_cfg
        else:
            self.color_cfg = EasyDict(
                enabled=True, img_w=640, img_h=480, img_format=rs.format.rgb8, fps=30
            )

        if depth_cfg is not None:
            self.depth_cfg = depth_cfg
        else:
            self.depth_cfg = EasyDict(
                enabled=False, img_w=640, img_h=480, img_format=rs.format.z16, fps=30
            )

        # TODO: Implement getting point clouds
        if pc_cfg is not None:
            self.pc_cfg = pc_cfg
        else:
            self.pc_cfg = EasyDict(enabled=False)

        if not (self.color_cfg.enabled or self.depth_cfg.enabled or self.pc_cfg.enabled):
            raise ValueError

        camera_config = EasyDict(
            enable_color=self.color_cfg.enabled,
            enable_depth=self.depth_cfg.enabled,
            enable_pc=self.pc_cfg.enabled,
            color_cfg=self.color_cfg,
            depth_cfg=self.depth_cfg,
            pc_cfg=self.pc_cfg,
        )
        self.camera = RSCameraWorker(
            camera_config=camera_config,
            device_id=device_id,
            thread_safe=False,
        )

    def start(self):
        self.camera.start()

    def get_last_obs(self):
        """
        Get last observation from camera
        """
        if self.camera.last_obs is None or self.camera.last_obs == {}:
            return None
        else:
            self.last_obs = self.camera.last_obs
            return self.last_obs

    def close(self):
        self.camera.halt()

    def get_camera_intrinsics(self):
        return self.camera.get_intrinsics()

    def get_depth_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("depth", mode=mode)
        return intrinsics

    def get_color_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("color", mode=mode)
        return intrinsics

    def get_color_distortion(self, mode=None):
        distortion = self.camera.get_distortion("color")
        return distortion
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor_id",
        type=int,
        default=9583,
    )
    parser.add_argument(
        "--product_id",
        type=int,
        default=50741,
    )
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-pose-controller.yml"
    )
    parser.add_argument(
        "--task_name", type=str, required=True
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Numeric task label to store; required if task_name is not an integer.",
    )
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--folder", type=str, default="/home/czhpc/deoxys_codebase/deoxys_control/deoxys/data_collection/")
    args = parser.parse_args()

    if args.task_id is None:
        try:
            args.task_id = int(args.task_name)
        except Exception as exc:
            raise ValueError(
                "--task_id must be provided when --task_name is not an integer"
            ) from exc

    args.folder = Path(f"{os.path.join(args.folder, args.task_name)}")
    return args

def main():
    args = parse_args()

    args.folder.mkdir(parents=True, exist_ok=True)

    experiment_id = 0
    logger.info(f"Saving to {args.folder}")
    for path in args.folder.glob("run*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1
    folder = str(args.folder / f"run{experiment_id}")
    os.makedirs(folder, exist_ok=True)

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id, device_index=0)
    device.start_control()

    # Franka Interface
    if args.interface_cfg[0] != "/":
        config_path = os.path.join(config_root, args.interface_cfg)
    else:
        config_path = args.interface_cfg
    robot_interface = FrankaInterface(config_path)

    controller_type = args.controller_type
    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    # If controller_cfg is different from user's specified controller
    # type, throw a warning and fall back to defautl config
    # print(controller_cfg.controller_type, controller_type)
    if controller_cfg.controller_type != controller_type:
        logger.warn("Config specification mismatched, using default controller config")
        controller_cfg = get_default_controller_config(controller_type)
        
    # camera_ids = [250122073979, 332522077725]
    # camera_ids = [332522077725, 244222071930]
    camera_ids = [250122073979]
    cr_interfaces = {}
    for camera_id in camera_ids:
        cr_interface = RSInterface(device_id=camera_id)
        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    data = {
        "action": [],
        "ee_states": [], 
        "joint_states": [], 
        "gripper_states": [],
        "task": [],
    }

    i = 0
    start = False
    previous_state_dict = None
    
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]

    # This is for varying initialization of joints a little bit to
    # increase data variation.
    reset_joint_positions = [
        e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        for e in reset_joint_positions
    ]

    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    goal_joint_positions = reset_joint_positions
    # goal_joint_positions = last_q.tolist()
    action = goal_joint_positions + [-1.0]

    while True:
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=YamlConfig(config_root + f"/joint-position-controller.yml").as_easydict(),
        )
        if len(robot_interface._state_buffer) > 0:
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

            if (
                np.max(
                    np.abs(
                        np.array(robot_interface._state_buffer[-1].q)
                        - np.array(goal_joint_positions)
                    )
                )
                < 1e-3
            ):
                break
        

    time.sleep(0.1)
    
    print("Start data collection...")
    while i < 4000:
        step_start_time = time.time()  # 记录每一步的开始时间
        # logger.info(i)
        i += 1
        start_time = time.time_ns()
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if action is None:
            break
        
        # set unused orientation dims to 0
        if controller_type == "OSC_YAW":
            action[3:5] = 0.0
        elif controller_type == "OSC_POSITION":
            action[3:6] = 0.0
        elif controller_type == "JOINT_IMPEDANCE":
            action[:7] = robot_interface.last_q.tolist()
        if i % 50 == 0:
            logger.info(f"step={i} start={start} action={np.round(action, 4)} state_buf={len(robot_interface._state_buffer)} last_q={np.round(robot_interface.last_q, 4)}")
        
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

        if len(robot_interface._state_buffer) == 0:
            continue

        last_state = robot_interface._state_buffer[-1]
        last_gripper_state = robot_interface._gripper_state_buffer[-1]
        if np.linalg.norm(action[:-1]) < 1e-3 and not start:
            # logger.info("Skipping storing zero action")
            continue

        start = True
        
        # Record data
        current_step = len(data["action"])  # Get current step number
        # Store action with an extra success flag dim (0/1).
        # Default to 0 during collection; we'll set the last step(s) to 1 after the loop.
        stored_action = list(np.asarray(action).reshape(-1)) + [0.0]
        data["action"].append(stored_action)
        state_dict = {
            "ee_states": np.array(last_state.O_T_EE),
            "joint_states": np.array(last_state.q),
            "gripper_states": np.array(last_gripper_state.width),
        }
        
        if previous_state_dict is not None:
            for proprio_key in state_dict.keys():
                proprio_state = state_dict[proprio_key]
                if np.sum(np.abs(proprio_state)) <= 1e-6:
                    proprio_state = previous_state_dict[proprio_key]
                state_dict[proprio_key] = np.copy(proprio_state)
        for proprio_key in state_dict.keys():
            data[proprio_key].append(state_dict[proprio_key])

        # Store per-timestep numeric task label (will be saved as (1, T) at the end)
        data["task"].append(int(args.task_id))
        
        previous_state_dict = state_dict

        # Save images immediately
        for camera_id in camera_ids:
            color_image = cr_interfaces[camera_id].get_last_obs()["color"]
            image_filename = os.path.join(folder, f"camera_{camera_id}_step_{current_step:06d}.png")
            iio.imwrite(image_filename, color_image)

        end_time = time.time_ns()
        # print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

        # Control the loop rate to match camera FPS
        step_duration = time.time() - step_start_time
        target_duration = 1.0 / 5.0  # 5 FPS
        if i % 50 == 0:
            logger.info(f"Step duration: {step_duration:.4f}s, Target duration: {target_duration:.4f}s")
        if step_duration < target_duration:
            time.sleep(target_duration - step_duration)

    # Save data
    os.makedirs(folder, exist_ok=True)

    # Mark the last 3 stored actions as success.
    # success flag is the last dim of stored_action.
    n_actions = len(data["action"])
    if n_actions > 0:
        for idx in range(max(0, n_actions - 3), n_actions):
            data["action"][idx][-1] = 1.0

    with open(f"{folder}/config.json", "w") as f:
        config_dict = {
            "controller_cfg": dict(controller_cfg),
            "controller_type": controller_type,
            "task_name": args.task_name,
            "task_id": int(args.task_id),
        }
        # import pdb; pdb.set_trace()
        json.dump(config_dict, f)
        np.savez(f"{folder}/testing_demo_action", data=np.array(data["action"]))
        np.savez(f"{folder}/testing_demo_ee_states", data=np.array(data["ee_states"]))
        np.savez(
            f"{folder}/testing_demo_joint_states", data=np.array(data["joint_states"])
        )
        np.savez(
            f"{folder}/testing_demo_gripper_states",
            data=np.array(data["gripper_states"]),
        )

        # Save task labels as shape (1, T)
        T = len(data["task"])
        task_arr = np.array(data["task"], dtype=np.int64).reshape(1, T)
        np.savez(f"{folder}/testing_demo_task", data=task_arr)
    
    # breakpoint()
    
    # Load and save camera data from saved images
    for camera_id in camera_ids:
        images = []
        total_steps = len(data["action"])
        for step in range(total_steps):
            image_path = os.path.join(folder, f"camera_{camera_id}_step_{step:06d}.png")
            if os.path.exists(image_path):
                image = iio.imread(image_path)
                images.append(image)
        np.savez(
            f"{folder}/testing_demo_camera_{camera_id}",
            data=np.array(images),
        )
        cr_interfaces[camera_id].close()
    robot_interface.close()
    print("Total length of the trajectory: ", len(data["action"]))

    valid_input = False
    while not valid_input:
        try:
            save = input("Save or not? (enter 0 or 1)")
            save = bool(int(save))
            valid_input = True
        except:
            pass
    if not save:
        import shutil

        shutil.rmtree(f"{folder}")

    # Always release SpaceMouse to avoid stale daemon thread across runs
    try:
        device.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()