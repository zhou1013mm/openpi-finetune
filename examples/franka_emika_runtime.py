import dataclasses
import logging
import time
from typing import Optional

import numpy as np
import tyro

from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

# Add deoxys to path (repo local)
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEOXYS_ROOT = (_REPO_ROOT.parent / "deoxys_control" / "deoxys").resolve()
if not _DEOXYS_ROOT.exists():
    raise RuntimeError(
        f"deoxys_control not found at {_DEOXYS_ROOT}. "
        "Clone deoxys_control as a sibling of openpi-finetune."
    )
if str(_DEOXYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEOXYS_ROOT))

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclasses.dataclass
class Args:
    # Task / prompt
    task: str = "clean_cook"

    # Policy server
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    api_key: Optional[str] = None

    # Action chunking
    action_horizon: int = 10

    # Control
    control_hz: float = 15.0
    joint_velocity_scale: float = 0.5
    controller_type: str = "JOINT_POSITION"
    interface_cfg: str = "charmander.yml"
    controller_cfg: str = "joint-position-controller.yml"

    # Camera (serial/device id). If numeric, treated as OpenCV index.
    exterior_camera_id: str = "33252207725"
    # Optional extra cameras (OpenCV indices). Use -1 to disable.
    exterior_cam_2: int = -1
    wrist_cam: int = -1

    # Runtime
    max_hz: float = 15.0
    num_episodes: int = 1
    max_episode_steps: int = 600


def _open_camera(device: str | int):
    if isinstance(device, int) and device < 0:
        return None
    if cv2 is None:
        raise RuntimeError("OpenCV is required for camera capture but is not installed.")
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device {device}")
    return cap


def _read_camera(cap, fallback_shape=(224, 224, 3)) -> np.ndarray:
    if cap is None:
        return np.zeros(fallback_shape, dtype=np.uint8)
    ret, frame = cap.read()
    if not ret:
        return np.zeros(fallback_shape, dtype=np.uint8)
    # BGR -> RGB
    frame = frame[:, :, ::-1]
    return frame


class FrankaDeoxysEnvironment(_environment.Environment):
    def __init__(self, args: Args) -> None:
        self._args = args
        self._robot = FrankaInterface(
            config_root + f"/{args.interface_cfg}",
            control_freq=args.control_hz,
            use_visualizer=False,
        )
        self._controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

        ext1_device: str | int = args.exterior_camera_id
        if args.exterior_camera_id.isdigit():
            ext1_device = int(args.exterior_camera_id)
        self._cam_ext_1 = _open_camera(ext1_device)
        self._cam_ext_2 = _open_camera(args.exterior_cam_2)
        self._cam_wrist = _open_camera(args.wrist_cam)

        self._episode_steps = 0

    def reset(self) -> None:
        self._robot.reset()
        self._episode_steps = 0

    def is_episode_complete(self) -> bool:
        if self._args.max_episode_steps > 0 and self._episode_steps >= self._args.max_episode_steps:
            return True
        return False

    def get_observation(self) -> dict:
        # Wait until we have state
        while not self._robot.received_states:
            time.sleep(0.01)

        joint_position = self._robot.last_q
        if joint_position is None:
            joint_position = np.zeros(7)

        gripper_position = self._robot.last_gripper_q
        if gripper_position is None:
            gripper_position = np.array([0.0])
        else:
            gripper_position = np.array([gripper_position])

        ext_1 = _read_camera(self._cam_ext_1)
        ext_2 = _read_camera(self._cam_ext_2)
        wrist = _read_camera(self._cam_wrist)

        ext_1 = image_tools.resize_with_pad(image_tools.convert_to_uint8(ext_1), 224, 224)
        ext_2 = image_tools.resize_with_pad(image_tools.convert_to_uint8(ext_2), 224, 224)
        wrist = image_tools.resize_with_pad(image_tools.convert_to_uint8(wrist), 224, 224)

        return {
            "observation/exterior_image_1_left": ext_1,
            "observation/exterior_image_2_left": ext_2,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": np.asarray(joint_position, dtype=np.float32),
            "observation/gripper_position": np.asarray(gripper_position, dtype=np.float32),
            "prompt": self._args.task,
        }

    def apply_action(self, action: dict) -> None:
        self._episode_steps += 1

        action_vec = action.get("actions")
        if action_vec is None:
            action_vec = action.get("action")
        if action_vec is None:
            raise ValueError("Policy action dict must contain 'actions' or 'action'.")

        action_vec = np.asarray(action_vec).reshape(-1)
        if action_vec.shape[0] < 7:
            raise ValueError(f"Expected at least 7 action dims, got {action_vec.shape[0]}")

        # Convert normalized joint velocity to joint position target
        current_q = self._robot.last_q
        if current_q is None:
            current_q = np.zeros(7)

        joint_vel = action_vec[:7] * self._args.joint_velocity_scale
        target_q = current_q + joint_vel * (1.0 / self._args.control_hz)

        # Gripper: >0.5 grasp, else open
        gripper_cmd = 1.0 if action_vec[-1] > 0.5 else -1.0

        ctrl_action = target_q.tolist() + [gripper_cmd]
        self._robot.control(
            controller_type=self._args.controller_type,
            action=ctrl_action,
            controller_cfg=self._controller_cfg,
        )


def main(args: Args) -> None:
    env = FrankaDeoxysEnvironment(args)

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
        api_key=args.api_key,
    )

    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=policy,
            action_horizon=args.action_horizon,
        )
    )

    runtime = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
