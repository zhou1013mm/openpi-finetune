#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run π₀.₅ DROID inference on a Franka arm via Deoxys.

This script mirrors the RDT inference flow but uses the π₀.₅ model.
It:
- boots RealSense cameras via RSInterface
- reads Franka state via Deoxys FrankaInterface
- formats observations and feeds them to the π₀.₅ policy
- executes predicted actions on the robot via OSC_POSE controller
- saves camera images asynchronously

Usage:
    python examples/franka_emika_runtime.py \
        --checkpoint_dir checkpoints/pi05_droid_finetune/clean_cook_jointpos/9999 \
        --train_config pi05_droid_finetune \
        --task clean_cook \
        --camera_ids "[332522077725]" \
"""

import dataclasses
import datetime
import logging
import queue
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tyro

import sys
sys.path.append("/home/czhpc/deoxys_codebase/deoxys_control/deoxys")

# Add deoxys to path (repo local)
_REPO_ROOT = Path(__file__).resolve().parents[1]
# _DEOXYS_ROOT = (_REPO_ROOT.parent / "deoxys_control" / "deoxys").resolve()
# if not _DEOXYS_ROOT.exists():
#     raise RuntimeError(
#         f"deoxys_control not found at {_DEOXYS_ROOT}. "
#         "Clone deoxys_control as a sibling of openpi-finetune."
#     )
# if str(_DEOXYS_ROOT) not in sys.path:
#     sys.path.insert(0, str(_DEOXYS_ROOT))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import image_tools

try:
    from PIL import Image
except ImportError:
    Image = None


def _require_deoxys():
    """Lazy import deoxys modules (avoid blocking on startup)."""
    try:
        from deoxys import config_root
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig
        from deoxys.utils.log_utils import get_deoxys_example_logger
        try:
            from spacemouse_collection_clean_table import RSInterface
        except ImportError:
            RSInterface = None
        return config_root, FrankaInterface, YamlConfig, get_deoxys_example_logger, RSInterface
    except Exception as e:
        raise RuntimeError(
            "Deoxys imports failed. Ensure you're on the robot machine with Deoxys installed. "
            f"Original error: {e}"
        )


@dataclasses.dataclass
class Args:
    # Task / prompt
    task: str = "clean_cook"

    # Policy
    checkpoint_dir: str = "checkpoints/pi05_droid_finetune/clean_cook_jointpos/9999"
    train_config: str = "pi05_droid_finetune"
    pytorch_device: str = "cuda"

    # Control
    control_hz: float = 5.0
    controller_type: str = "JOINT_POSITION"
    interface_cfg: str = "charmander.yml"
    controller_cfg: str = "joint-position-controller.yml"

    # Action interpretation
    # "joint_position" -> actions are absolute joint positions
    # "joint_velocity" -> actions are joint velocities (integrated with control_hz)
    action_space: str = "joint_position"
    # Threshold (meters) for mapping gripper position to open/close command
    gripper_open_threshold: float = 0.04

    # Camera (RealSense device serials)
    camera_ids: str = "[332522077725]"
    num_cameras: int = 1

    # Runtime
    max_hz: float = 5.0
    max_duration: float = 1200.0
    steps_per_inference: int = 8

    # Rate limiting (enforce control_hz)
    enforce_control_hz: bool = True

    # Logging
    save_root: str = ""  # If empty, auto-generate from timestamp
    
    # Test mode (skip robot/camera, just test model)
    test_mode: bool = False

    # Logging
    save_root: str = ""  # If empty, auto-generate from timestamp


class _AsyncImageSaver:
    """Save images asynchronously to avoid blocking inference."""

    def __init__(self, out_dir: str, max_queue: int = 256, image_format: str = "jpg"):
        self.out_dir = out_dir
        self.image_format = image_format.lower()
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0

        import os

        os.makedirs(self.out_dir, exist_ok=True)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, name="async_image_saver", daemon=True)
        self._thread.start()

    def submit(self, step_idx: int, images_by_key: Dict[str, Optional[np.ndarray]]) -> None:
        """Non-blocking: if queue is full, drop the frame."""
        if self._stop.is_set():
            return

        ts_ms = int(time.time() * 1000)
        for key, img in images_by_key.items():
            if img is None:
                continue
            item = (step_idx, ts_ms, key, np.asarray(img).copy())
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                return

    def stop(self, drain: bool = True, timeout: float = 5.0) -> None:
        self._stop.set()
        if drain:
            t0 = time.time()
            while (not self._q.empty()) and (time.time() - t0 < timeout):
                time.sleep(0.01)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def stats(self) -> Dict[str, Any]:
        return {"dropped": self._dropped, "queued": self._q.qsize(), "out_dir": self.out_dir}

    def _worker(self) -> None:
        while True:
            if self._stop.is_set() and self._q.empty():
                break
            try:
                step_idx, ts_ms, key, img = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if Image is None:
                    continue
                pil = Image.fromarray(img)
                import os

                subdir = os.path.join(self.out_dir, key)
                os.makedirs(subdir, exist_ok=True)
                fname = f"step_{step_idx:06d}_t{ts_ms}.{self.image_format}"
                fpath = os.path.join(subdir, fname)
                if self.image_format in {"jpg", "jpeg"}:
                    pil.save(fpath, quality=95)
                else:
                    pil.save(fpath)
            except Exception:
                pass
            finally:
                self._q.task_done()


class _KeyboardQuitter:
    """Sets an event when user presses 'q' (or types 'q' + Enter)."""

    def __init__(self):
        self.quit_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="keyboard_quit", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            if sys.stdin is None:
                return
            if sys.stdin.isatty():
                import select

                import termios
                import tty

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not self.quit_event.is_set():
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if not r:
                            continue
                        ch = sys.stdin.read(1)
                        if ch and ch.lower() == "q":
                            self.quit_event.set()
                            return
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            else:
                for line in sys.stdin:
                    if line.strip().lower() == "q":
                        self.quit_event.set()
                        return
        except Exception:
            return


def _parse_camera_ids(camera_ids_raw: str) -> List[int]:
    """Parse camera IDs from string like '[1,2,3]' or '1,2,3'."""
    import ast

    s = str(camera_ids_raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        parsed = ast.literal_eval(s)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"--camera_ids must be a list, e.g. [1,2]")
        return [int(x) for x in parsed]
    parts = [p for p in s.replace(",", " ").split() if p]
    return [int(p) for p in parts]


def build_cameras(camera_ids: List[int], RSInterface) -> Dict[str, Any]:
    """Build RealSense camera dict by serial ID with timeout protection."""
    if RSInterface is None:
        logging.warning("RSInterface not available; using dummy cameras")
        return {f"cam_{i}": None for i in range(len(camera_ids))}

    cam_by_key: Dict[str, Any] = {}
    for i, serial in enumerate(camera_ids):
        try:
            logging.info(f"Opening camera {i} (serial={serial})...")
            cam = RSInterface(device_id=int(serial))
            logging.info(f"Camera {i} opened, starting stream...")
            cam.start()
            logging.info(f"Camera {i} stream started.")
            cam_by_key[f"cam_{i}"] = cam
        except Exception as e:
            logging.warning(f"Failed to open camera {serial}: {e}; will use None")
            cam_by_key[f"cam_{i}"] = None
    return cam_by_key


def close_cameras(cam_by_key: Dict[str, Any]):
    """Close all camera connections."""
    for cam in cam_by_key.values():
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass


def fetch_camera_images(cam_by_key: Dict[str, Any], fallback_shape: tuple = (224, 224, 3)) -> Dict[str, np.ndarray]:
    """Fetch latest images from all cameras."""
    imgs: Dict[str, np.ndarray] = {}
    for k, cam in cam_by_key.items():
        if cam is None:
            imgs[k] = np.zeros(fallback_shape, dtype=np.uint8)
        else:
            try:
                last = cam.get_last_obs()
                if last is not None and "color" in last:
                    img = np.asarray(last["color"], dtype=np.uint8)
                    imgs[k] = image_tools.resize_with_pad(image_tools.convert_to_uint8(img), 224, 224)
                else:
                    imgs[k] = np.zeros(fallback_shape, dtype=np.uint8)
            except Exception:
                imgs[k] = np.zeros(fallback_shape, dtype=np.uint8)
    return imgs


def fetch_franka_state(robot_interface) -> Dict[str, np.ndarray]:
    """Fetch current Franka state (joint positions, gripper)."""
    if len(robot_interface._state_buffer) == 0 or len(robot_interface._gripper_state_buffer) == 0:
        raise RuntimeError("Robot state buffer empty")

    q = np.asarray(robot_interface._state_buffer[-1].q, dtype=np.float32)
    grip_width = float(robot_interface._gripper_state_buffer[-1].width)

    return {
        "joint_position": q,
        "gripper_position": np.array([grip_width], dtype=np.float32),
    }


def _map_gripper_action(gripper_position: float, open_threshold: float) -> float:
    """Map gripper position (width) to Deoxys open/close command.

    Deoxys expects: action < 0 => open, action >= 0 => close.
    """
    return -1.0 if gripper_position >= open_threshold else 1.0


def reset_robot_to_home(robot_interface, config_root: Path, logger, YamlConfig) -> None:
    """Reset robot to home position using JOINT_POSITION controller."""
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]

    # Add slight random variation
    reset_joint_positions = [
        e + float(np.clip(np.random.randn() * 0.005, -0.005, 0.005)) for e in reset_joint_positions
    ]

    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    action = reset_joint_positions + [-1.0]
    joint_pos_cfg = YamlConfig(str(config_root / "joint-position-controller.yml")).as_easydict()

    logger.info("Resetting to home position...")
    while True:
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=joint_pos_cfg,
        )
        if len(robot_interface._state_buffer) > 0:
            if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-3:
                break
    time.sleep(0.5)
    logger.info("Robot at home.")


def apply_action(robot_interface, controller_type: str, controller_cfg, action_vec: np.ndarray, logger):
    """Send action to robot."""
    action_vec = np.asarray(action_vec, dtype=np.float64).reshape(-1)
    try:
        robot_interface.control(controller_type=controller_type, action=action_vec.tolist(), controller_cfg=controller_cfg)
    except Exception as e:
        logger.error(f"Failed to apply action: {e}")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    
    # Lazy import deoxys (happens here, not at startup)
    print("Importing deoxys...")
    config_root, FrankaInterface, YamlConfig, get_deoxys_example_logger, RSInterface = _require_deoxys()
    print("Deoxys imported successfully.")
    config_root = Path(config_root)
    logger = get_deoxys_example_logger()
    logger.info("Deoxys logger initialized.")

    # Setup output directory
    if not args.save_root:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_root = f"outputs/franka_pi05_inference_{run_id}"
    images_out_dir = f"{args.save_root}/camera_images"

    logger.info(f"Saving outputs to: {args.save_root}")
    logger.info(f"Saving camera images to: {images_out_dir}")

    # Setup async workers
    image_saver = _AsyncImageSaver(out_dir=images_out_dir, max_queue=256, image_format="jpg")
    image_saver.start()

    quitter = _KeyboardQuitter()
    quitter.start()
    logger.info("Press 'q' to quit.")

    # Load policy from checkpoint
    checkpoint_path = (_REPO_ROOT / args.checkpoint_dir).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    logger.info(f"Loading policy from {checkpoint_path}...")
    train_cfg = _config.get_config(args.train_config)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        checkpoint_path,
        default_prompt=args.task,
        pytorch_device=args.pytorch_device,
    )
    logger.info("Policy loaded.")

    # Setup cameras
    camera_ids = _parse_camera_ids(args.camera_ids)
    cam_by_key = build_cameras(camera_ids, RSInterface)
    logger.info(f"Initialized {len(cam_by_key)} camera(s).")

    # Setup robot
    interface_cfg_path = args.interface_cfg
    if not interface_cfg_path.startswith("/"):
        interface_cfg_path = str(config_root / interface_cfg_path)
    robot_interface = FrankaInterface(interface_cfg_path)
    logger.info("Franka interface initialized.")

    controller_cfg_obj = YamlConfig(str(config_root / args.controller_cfg)).as_easydict()
    logger.info(f"Using controller: {args.controller_type}")

    target_dt = 1.0 / float(args.control_hz) if args.control_hz and args.control_hz > 0 else 0.0
    action_space = str(args.action_space).strip().lower()

    obs_window: deque = deque(maxlen=2)
    t_start = time.monotonic()

    try:
        reset_robot_to_home(robot_interface, config_root, logger, YamlConfig)

        # Warm up observation buffer
        logger.info("Warming up observations...")
        while len(obs_window) < 2:
            imgs = fetch_camera_images(cam_by_key)
            image_saver.submit(step_idx=-1, images_by_key=imgs)
            state = fetch_franka_state(robot_interface)
            obs_window.append({"images": imgs, "state": state})
            time.sleep(0.05)

        logger.info("Starting control loop...")
        step_idx = 0

        while (time.monotonic() - t_start) < args.max_duration:
            if quitter.quit_event.is_set():
                logger.info("Quit requested. Exiting control loop...")
                break

            imgs = fetch_camera_images(cam_by_key)
            image_saver.submit(step_idx=step_idx, images_by_key=imgs)
            state = fetch_franka_state(robot_interface)
            obs_window.append({"images": imgs, "state": state})

            if len(obs_window) < 2:
                time.sleep(0.01)
                continue

            curr = obs_window[-1]

            # Prepare observation dict for policy
            cam_0 = curr["images"].get("cam_0", np.zeros((224, 224, 3), dtype=np.uint8))
            cam_blank = np.zeros_like(cam_0)
            cam_1 = cam_blank
            cam_2 = cam_0
            obs = {
                "observation/exterior_image_1_left": cam_0,
                "observation/exterior_image_2_left": cam_1,
                "observation/wrist_image_left": cam_2,
                "observation/joint_position": curr["state"]["joint_position"],
                "observation/gripper_position": curr["state"]["gripper_position"],
                "prompt": args.task,
            }

            # Inference
            try:
                result = policy.infer(obs)
                action_seq = result["actions"]  # Shape: (horizon, action_dim)
            except Exception as e:
                logger.error(f"Policy inference failed: {e}")
                break

            # Execute actions
            k_exec = max(1, min(args.steps_per_inference, int(action_seq.shape[0])))
            for k in range(k_exec):
                if quitter.quit_event.is_set():
                    logger.info("Quit requested. Exiting control loop...")
                    raise KeyboardInterrupt

                step_t0 = time.monotonic()
                action = np.asarray(action_seq[k], dtype=np.float64).reshape(-1)
                if action.shape[0] < 8:
                    raise ValueError(f"Expected action dim >= 8 (7 joints + gripper), got {action.shape[0]}")

                joint_cmd = action[:7]
                gripper_val = float(action[7])

                if action_space == "joint_velocity":
                    current_q = curr["state"]["joint_position"]
                    target_q = current_q + joint_cmd * target_dt
                elif action_space == "joint_position":
                    target_q = joint_cmd
                else:
                    raise ValueError(
                        f"Unknown action_space '{args.action_space}'. Use 'joint_position' or 'joint_velocity'."
                    )

                gripper_cmd = _map_gripper_action(gripper_val, args.gripper_open_threshold)
                full_action = np.concatenate([np.asarray(target_q, dtype=np.float64), [gripper_cmd]], axis=0)
                apply_action(robot_interface, args.controller_type, controller_cfg_obj, full_action, logger)

                if args.enforce_control_hz and target_dt > 0:
                    elapsed = time.monotonic() - step_t0
                    sleep_dt = target_dt - elapsed
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)

            step_idx += 1
            if step_idx % 100 == 0:
                logger.info(f"Step {step_idx}, elapsed time: {(time.monotonic() - t_start) / 60:.1f} min")

    finally:
        logger.info("Shutting down...")
        try:
            image_saver.stop(drain=True, timeout=5.0)
            logger.info(f"Image saver stats: {image_saver.stats()}")
        except Exception:
            pass
        close_cameras(cam_by_key)
        try:
            robot_interface.close()
        except Exception:
            pass
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
