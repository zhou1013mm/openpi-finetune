#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run RDT inference on a Franka arm via Deoxys.

This script mirrors the Deoxys Franka eval flow (see eval_deoxys_franka_cube.py)
but swaps in the RDT model wrapper. It:
- boots RealSense cameras via RSInterface
- reads Franka state via Deoxys FrankaInterface
- formats state/images into the unified 128-D action/state space expected by RDT
- runs RDT to predict delta end-effector command (OSC pose)
- sends the command to the robot via Deoxys

Usage (example):
    python -m scripts.franka_rdt_deoxys_inference \
        --pretrained_model_name_or_path checkpoints/rdt-finetune-170m/checkpoint-22000 \
        --lang_embeddings_path datasets/cube_touch_lang_embed.pt \
        --interface_cfg charmander.yml \
        --controller_type OSC_POSE --controller_cfg osc-pose-controller.yml \
        --camera_ids "[250122073979]"

Notes:
- This repo's default fine-tune config uses `common.num_cameras=1` and `common.img_history_size=1`,
    so training feeds exactly ONE external RGB image per step. This script follows the config and
    feeds exactly `img_history_size * num_cameras` images at inference.
- If you later change the config to use multiple cameras and/or history, the script will pad
    missing cameras with background images.
- Reset uses JOINT_POSITION to go to a fixed home pose. Control uses OSC_POSE by default.
- State/action mapping follows `data/npz_vla_dataset.py`:
  - state dims (10): right_eef_pos(3) + right_eef_rot6d(6) + right_gripper_open(1)
  - action dims (7): right_eef_vel(3) + right_eef_angular_vel(3) + right_gripper_open_vel(1)
"""
import argparse
import sys
sys.path.append('/home/czhpc/deoxys_codebase/deoxys_control/deoxys')

import ast
import datetime
import os
import queue
import select
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.rdt_runner import RDTRunner


class _AsyncImageSaver:
    def __init__(self, out_dir: str, max_queue: int = 256, image_format: str = "jpg"):
        self.out_dir = out_dir
        self.image_format = image_format.lower()
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0

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
            # Copy to decouple from upstream buffers
            item = (step_idx, ts_ms, key, np.asarray(img).copy())
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                return

    def stop(self, drain: bool = True, timeout: float = 5.0) -> None:
        self._stop.set()
        if drain:
            # Wait briefly for queue to empty
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
                # RealSense typically yields HxWx3 uint8.
                pil = Image.fromarray(img)
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
    """Sets an event when user presses 'q' (or types 'q' + Enter).

    - If stdin is a TTY, reads single chars without blocking.
    - Otherwise falls back to line-based input.
    """

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
                # Non-blocking single-char read on Linux.
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
                # Line-based fallback
                for line in sys.stdin:
                    if line.strip().lower() == "q":
                        self.quit_event.set()
                        return
        except Exception:
            return


def _load_lang_embeddings(path: str, device: str) -> tuple[torch.Tensor, Optional[str]]:
    """Load pre-encoded language embeddings.

    This repo has two common formats:
    1) A plain tensor saved by scripts/encode_lang_tensor.py with shape (seq_len, hidden_dim)
    2) A dict saved by scripts/encode_lang_batch.py with keys like {"instruction", "embeddings"}

    Returns:
      text_embeds: torch.Tensor with shape (B, L, D)
      instruction: optional instruction string (if present in file)
    """
    obj = torch.load(path, map_location="cpu")

    instruction: Optional[str] = None
    text_embeds: Any

    if isinstance(obj, dict):
        if "embeddings" in obj:
            text_embeds = obj["embeddings"]
            instruction = obj.get("instruction")
        else:
            # Sometimes users save {instruction_str: tensor}.
            # Pick the first tensor value if possible.
            picked = None
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    picked = (k, v)
                    break
            if picked is None:
                raise ValueError(
                    f"Unsupported lang embedding dict in {path}. "
                    f"Expected key 'embeddings' or a mapping to a Tensor. Keys={list(obj.keys())[:10]}"
                )
            instruction = str(picked[0])
            text_embeds = picked[1]
    else:
        text_embeds = obj

    if not isinstance(text_embeds, torch.Tensor):
        raise TypeError(f"Unsupported lang embedding type: {type(text_embeds)} from {path}")

    # Normalize to (B, L, D)
    if text_embeds.dim() == 2:
        text_embeds = text_embeds.unsqueeze(0)
    elif text_embeds.dim() == 3:
        pass
    else:
        raise ValueError(
            f"Expected lang embeddings to have 2 or 3 dims, got shape {tuple(text_embeds.shape)} from {path}"
        )

    # Keep it on CPU here; it will be moved/cast in the model step.
    return text_embeds.contiguous(), instruction


# =======================
# Deoxys / vision helpers
# =======================

def _require_deoxys():
    try:
        from deoxys import config_root
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig
        from deoxys.utils.config_utils import get_default_controller_config
        from deoxys.utils.log_utils import get_deoxys_example_logger
        from spacemouse_collection_clean_table import RSInterface

        return config_root, FrankaInterface, YamlConfig, get_default_controller_config, RSInterface, get_deoxys_example_logger
    except Exception as e:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "Deoxys / deoxys_vision imports failed. Run on the robot machine with Deoxys installed. "
            f"Original error: {e}"
        )


def _parse_camera_ids(camera_ids_raw: str) -> List[int]:
    s = str(camera_ids_raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        parsed = ast.literal_eval(s)
        if not isinstance(parsed, (list, tuple)):
            raise argparse.ArgumentTypeError("--camera_ids must be a list, e.g. [1,2]")
        return [int(x) for x in parsed]
    parts = [p for p in s.replace(",", " ").split() if p]
    return [int(p) for p in parts]


# =======================
# RDT Franka wrapper
# =======================

# NPZ dataset mapping (used by `data/npz_vla_dataset.py`)
FRANKA_EEF_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_z"],
    STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
    STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
    STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
    STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
    STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
    STATE_VEC_IDX_MAPPING["right_eef_angle_5"],
    STATE_VEC_IDX_MAPPING["right_gripper_open"],
]
FRANKA_EEF_ACTION_INDICES = [
    STATE_VEC_IDX_MAPPING["right_eef_vel_x"],
    STATE_VEC_IDX_MAPPING["right_eef_vel_y"],
    STATE_VEC_IDX_MAPPING["right_eef_vel_z"],
    STATE_VEC_IDX_MAPPING["right_eef_angular_vel_roll"],
    STATE_VEC_IDX_MAPPING["right_eef_angular_vel_pitch"],
    STATE_VEC_IDX_MAPPING["right_eef_angular_vel_yaw"],
    STATE_VEC_IDX_MAPPING["right_gripper_open_vel"],
]


class FrankaRDTModel:
    def __init__(
        self,
        config: dict,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        control_frequency: int = 5,
        pretrained: Optional[str] = None,
        vision_encoder_name_or_path: str = "google/siglip-so400m-patch14-384",
        load_finetuned_vision_encoder: bool = True,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.control_frequency = control_frequency

        self.image_processor, self.vision_model = self._get_vision_encoder(vision_encoder_name_or_path)
        if load_finetuned_vision_encoder:
            self._maybe_load_finetuned_vision_encoder(pretrained)
        self.policy = self._get_policy(pretrained)
        self.reset()

    def reset(self):
        self.policy.eval()
        self.vision_model.eval()
        self.policy = self.policy.to(self.device, dtype=self.dtype)
        self.vision_model = self.vision_model.to(self.device, dtype=self.dtype)

    def _get_vision_encoder(self, name_or_path: str):
        vision_encoder = SiglipVisionTower(vision_tower=name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def _maybe_load_finetuned_vision_encoder(self, pretrained: Optional[str]) -> None:
        """If training enabled --train_vision_encoder, its weights may be saved inside the
        checkpoint's pytorch_model.bin under the prefix `vision_encoder.*`.

        RDTRunner.from_pretrained() does not restore those extra keys (strict=False and
        RDTRunner doesn't define the submodule), so we load them explicitly here.
        """
        if pretrained is None:
            return
        if not os.path.isdir(pretrained):
            return

        ckpt_file = os.path.join(pretrained, "pytorch_model.bin")
        if not os.path.isfile(ckpt_file):
            return

        try:
            state = torch.load(ckpt_file, map_location="cpu")
        except Exception:
            return

        prefix = "vision_encoder."
        vision_sd = {k[len(prefix) :]: v for k, v in state.items() if isinstance(k, str) and k.startswith(prefix)}
        if len(vision_sd) == 0:
            return

        missing, unexpected = self.vision_model.load_state_dict(vision_sd, strict=False)
        print(
            f"Loaded finetuned vision encoder from {ckpt_file} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _get_policy(self, pretrained: Optional[str]):
        img_cond_len = (
            self.config["common"]["img_history_size"]
            * self.config["common"]["num_cameras"]
            * self.vision_model.num_patches
        )

        model = RDTRunner.from_pretrained(pretrained)
        return model

    def _format_joint_to_state(self, joints: torch.Tensor):
        raise NotImplementedError("Joint-space state is not used in this script.")

    def _unformat_action_to_joint(self, action: torch.Tensor):
        raise NotImplementedError("Joint-space action is not used in this script.")

    def _format_eef_to_state(self, eef_state: torch.Tensor):
        # eef_state: [B, N, 10] (pos3 + rot6d6 + gripper1)
        if eef_state.dim() == 1:
            eef_state = eef_state.unsqueeze(0)
        if eef_state.dim() == 2:
            eef_state = eef_state.unsqueeze(1)
        eef_state = eef_state.to(self.device, dtype=self.dtype)

        B, N, D = eef_state.shape
        if D != 10:
            raise ValueError(f"Expected eef_state dim 10, got {D}")

        state = torch.zeros(
            (B, N, self.config["model"]["state_token_dim"]),
            device=eef_state.device,
            dtype=eef_state.dtype,
        )
        state[:, :, FRANKA_EEF_STATE_INDICES] = eef_state

        # This is the `action_mask` expected by RDTRunner: mark valid action dimensions.
        action_mask = torch.zeros(
            (B, self.config["model"]["state_token_dim"]),
            device=eef_state.device,
            dtype=eef_state.dtype,
        )
        action_mask[:, FRANKA_EEF_ACTION_INDICES] = 1
        return state, action_mask

    def _unformat_action_to_osc7(self, action: torch.Tensor) -> torch.Tensor:
        # action: [B, horizon, 128] -> osc7 [B, horizon, 7]
        osc7 = action[:, :, FRANKA_EEF_ACTION_INDICES]
        return osc7

    @torch.no_grad()
    def step(self, proprio: torch.Tensor, images: List[Optional[np.ndarray]], text_embeds: torch.Tensor):
        device = self.device
        dtype = self.dtype

        background_color = np.array([int(x * 255) for x in self.image_processor.image_mean], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones(
            (self.image_processor.size["height"], self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color

        image_tensor_list = []
        for img in images:
            if img is None:
                pil_img = Image.fromarray(background_image)
            else:
                pil_img = Image.fromarray(img)
            if self.config["dataset"].get("image_aspect_ratio", "pad") == "pad":
                def expand2square(pil_img_in: Image.Image, bg_color):
                    w, h = pil_img_in.size
                    if w == h:
                        return pil_img_in
                    if w > h:
                        canvas = Image.new(pil_img_in.mode, (w, w), bg_color)
                        canvas.paste(pil_img_in, (0, (w - h) // 2))
                        return canvas
                    canvas = Image.new(pil_img_in.mode, (h, h), bg_color)
                    canvas.paste(pil_img_in, ((h - w) // 2, 0))
                    return canvas
                pil_img = expand2square(pil_img, tuple(int(x * 255) for x in self.image_processor.image_mean))
            pixel = self.image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
            image_tensor_list.append(pixel)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        eef_state = proprio.to(device).float()
        if eef_state.dim() == 1:
            eef_state = eef_state.unsqueeze(0)
        if eef_state.dim() == 2:
            eef_state = eef_state.unsqueeze(1)
        states, action_mask = self._format_eef_to_state(eef_state)
        ctrl_freqs = torch.tensor([self.control_frequency], device=device)

        text_embeds = text_embeds.to(device, dtype=dtype)

        traj = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=action_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
        )
        osc7 = self._unformat_action_to_osc7(traj).to(torch.float32)
        return osc7


# =======================
# Camera + robot helpers
# =======================

ALL_CAMERA_KEYS = ["cam_ext", "cam_right_wrist", "cam_left_wrist"]


def _select_camera_keys(num_cameras: int) -> List[str]:
    if num_cameras <= 0:
        raise ValueError(f"num_cameras must be >= 1, got {num_cameras}")
    if num_cameras > len(ALL_CAMERA_KEYS):
        raise ValueError(
            f"num_cameras={num_cameras} exceeds supported keys {ALL_CAMERA_KEYS}. "
            "Extend ALL_CAMERA_KEYS mapping if needed."
        )
    return ALL_CAMERA_KEYS[:num_cameras]


def build_cameras(camera_ids: List[int], camera_keys: List[str], RSInterface) -> Dict[str, Any]:
    cam_by_key: Dict[str, Any] = {}
    if len(camera_ids) < len(camera_keys):
        raise ValueError(
            f"Need at least {len(camera_keys)} camera_ids for {camera_keys}, got {len(camera_ids)}"
        )
    for key, serial in zip(camera_keys, camera_ids[: len(camera_keys)]):
        cam = RSInterface(device_id=int(serial))
        cam.start()
        cam_by_key[key] = cam
    return cam_by_key


def close_cameras(cam_by_key: Dict[str, Any]):
    for cam in cam_by_key.values():
        try:
            cam.close()
        except Exception:
            pass


def fetch_camera_images(cam_by_key: Dict[str, Any], camera_keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
    imgs: Dict[str, Optional[np.ndarray]] = {k: None for k in camera_keys}
    for k in camera_keys:
        cam = cam_by_key[k]
        last = cam.get_last_obs()
        if last is not None and "color" in last:
            imgs[k] = np.asarray(last["color"], dtype=np.uint8)
    return imgs


def fetch_franka_state(robot_interface) -> np.ndarray:
    if len(robot_interface._state_buffer) == 0 or len(robot_interface._gripper_state_buffer) == 0:
        raise RuntimeError("Robot state buffer empty")
    q = np.asarray(robot_interface._state_buffer[-1].q, dtype=np.float32)
    grip_width = float(robot_interface._gripper_state_buffer[-1].width)
    return np.concatenate([q, [grip_width]], axis=0)


def _ee16_to_pos_rot6d_single(ee16: np.ndarray) -> np.ndarray:
    """Convert Franka/libfranka O_T_EE flattened 4x4 to 9D pose.

    pose9 = [pos(3), rot6d(6)] where rot6d is the first two columns of R.
    Includes a heuristic transpose for common column-major flattening.
    """
    ee16 = np.asarray(ee16)
    if ee16.size != 16:
        raise ValueError(f"Expected ee16 size 16, got {ee16.size}")

    mat = ee16.reshape(4, 4).astype(np.float32)

    br_ok = np.isfinite(mat[3, 3]) and (abs(float(mat[3, 3]) - 1.0) < 1e-2)
    last_col_small = float(np.linalg.norm(mat[:3, 3])) < 1e-3
    last_row_large = float(np.linalg.norm(mat[3, :3])) > 1e-4
    if br_ok and last_col_small and last_row_large:
        mat = mat.T

    R = mat[:3, :3]
    pos = mat[:3, 3]
    rot6d = R[:, :2].reshape(6)
    return np.concatenate([pos, rot6d], axis=0).astype(np.float32)


def fetch_franka_eef_state(robot_interface) -> np.ndarray:
    if len(robot_interface._state_buffer) == 0:
        raise RuntimeError("Robot state buffer empty")
    if len(robot_interface._gripper_state_buffer) == 0:
        raise RuntimeError("Gripper state buffer empty")

    state = robot_interface._state_buffer[-1]
    ee16 = np.asarray(state.O_T_EE, dtype=np.float32)
    pose9 = _ee16_to_pos_rot6d_single(ee16)

    gripper_state = robot_interface._gripper_state_buffer[-1]
    gripper_width = np.asarray([float(gripper_state.width)], dtype=np.float32)
    return np.concatenate([pose9, gripper_width], axis=0)


def reset_robot_to_home(robot_interface, config_root: Path, logger, YamlConfig) -> None:
    # Copied (lightly) from eval_deoxys_franka_cube.py
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]

    # Vary initialization slightly to avoid exact repeats
    reset_joint_positions = [
        e + float(np.clip(np.random.randn() * 0.005, -0.005, 0.005)) for e in reset_joint_positions
    ]

    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    goal_joint_positions = reset_joint_positions
    # Deoxys JOINT_POSITION expects 8D: 7 joints + gripper
    action = goal_joint_positions + [-1.0]

    joint_pos_cfg = YamlConfig(str(config_root / "joint-position-controller.yml")).as_easydict()
    while True:
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=joint_pos_cfg,
        )
        if len(robot_interface._state_buffer) > 0:
            if (
                np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(goal_joint_positions)))
                < 1e-3
            ):
                break
    time.sleep(0.5)


# =======================
# Control loop
# =======================


def apply_action(robot_interface, controller_type: str, controller_cfg, action_vec: np.ndarray):
    action_vec = np.asarray(action_vec, dtype=np.float64).reshape(-1)
    robot_interface.control(controller_type=controller_type, action=action_vec.tolist(), controller_cfg=controller_cfg)


def main():
    parser = argparse.ArgumentParser(description="Franka Deoxys inference with RDT")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        required=True,
        help=(
            "RDT model source: either a HuggingFace model id (e.g. robotics-diffusion-transformer/rdt-1b) "
            "or a local checkpoint directory (e.g. checkpoints/rdt-finetune-1b/checkpoint-55000)."
        ),
    )
    parser.add_argument("--lang_embeddings_path", required=True)
    parser.add_argument("--config_path", default="configs/base.yaml")
    parser.add_argument("--ctrl_freq", type=int, default=5)
    parser.add_argument("--camera_ids", type=str, required=True, help="List or csv of RealSense serials")
    parser.add_argument("--interface_cfg", required=True, help="Deoxys Franka interface config (e.g. charmander.yml)")
    parser.add_argument(
        "--controller_type",
        default="OSC_POSE",
        help="Deoxys controller used during the control loop (reset always uses JOINT_POSITION).",
    )
    parser.add_argument("--controller_cfg", default="osc-pose-controller.yml")
    parser.add_argument("--steps_per_inference", type=int, default=4)
    parser.add_argument("--max_duration", type=float, default=1200.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no_load_finetuned_vision_encoder",
        action="store_true",
        help=(
            "Do not load vision encoder weights from the RDT checkpoint even if present; "
            "always use base google/siglip-so400m-patch14-384 weights."
        ),
    )
    args = parser.parse_args()

    import yaml

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join("outputs", f"franka_rdt_inference_{run_id}")
    images_out_dir = os.path.join(save_root, "camera_images")

    image_saver = _AsyncImageSaver(out_dir=images_out_dir, max_queue=256, image_format="jpg")
    image_saver.start()

    quitter = _KeyboardQuitter()
    quitter.start()
    print("Press 'q' to quit.")
    print(f"Saving camera images to: {images_out_dir}")

    num_cameras = int(cfg["common"].get("num_cameras", 1))
    img_history_size = int(cfg["common"].get("img_history_size", 1))
    camera_keys = _select_camera_keys(num_cameras)

    text_embeds, instruction = _load_lang_embeddings(args.lang_embeddings_path, device=args.device)
    if instruction is not None:
        print(f"Loaded instruction: {instruction}")
    else:
        print(f"Loaded lang embeddings: shape={tuple(text_embeds.shape)}")

    config_root, FrankaInterface, YamlConfig, get_default_controller_config, RSInterface, get_logger = _require_deoxys()
    config_root = Path(config_root)

    cam_ids = _parse_camera_ids(args.camera_ids)
    cam_by_key = build_cameras(cam_ids, camera_keys, RSInterface)

    interface_cfg_path = args.interface_cfg
    if not interface_cfg_path.startswith("/"):
        interface_cfg_path = str(config_root / interface_cfg_path)
    robot_interface = FrankaInterface(interface_cfg_path)

    controller_cfg_obj = YamlConfig(str(config_root / args.controller_cfg)).as_easydict()
    if getattr(controller_cfg_obj, "controller_type", args.controller_type) != args.controller_type:
        controller_cfg_obj = get_default_controller_config(args.controller_type)

    model = FrankaRDTModel(
        config=cfg,
        device=args.device,
        dtype=torch.bfloat16,
        control_frequency=args.ctrl_freq,
        pretrained=args.pretrained_model_name_or_path,
        vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
        load_finetuned_vision_encoder=(not args.no_load_finetuned_vision_encoder),
    )

    obs_window: deque = deque(maxlen=2)
    t_start = time.monotonic()
    logger = get_logger()

    try:
        print("Resetting robot to home (JOINT_POSITION)...")
        reset_robot_to_home(robot_interface, config_root, logger, YamlConfig)

        print("Warming up observations...")
        while len(obs_window) < 2:
            imgs = fetch_camera_images(cam_by_key, camera_keys)
            image_saver.submit(step_idx=-1, images_by_key=imgs)
            eef = fetch_franka_eef_state(robot_interface)
            obs_window.append({"state": torch.from_numpy(eef).float(), "images": imgs})
            time.sleep(0.05)

        print("Starting control loop...")
        target_dt = 1.0 / max(1.0, float(args.ctrl_freq))
        step_idx = 0
        while (time.monotonic() - t_start) < args.max_duration:
            if quitter.quit_event.is_set():
                print("Quit requested (q). Exiting control loop...")
                break

            imgs = fetch_camera_images(cam_by_key, camera_keys)
            image_saver.submit(step_idx=step_idx, images_by_key=imgs)
            eef = fetch_franka_eef_state(robot_interface)
            obs_window.append({"state": torch.from_numpy(eef).float(), "images": imgs})

            if len(obs_window) < 2:
                time.sleep(0.01)
                continue

            curr = obs_window[-1]
            # Follow training config: feed exactly img_history_size*num_cameras images.
            # Default fine-tune config in this repo: num_cameras=1, img_history_size=1 -> [cam_ext_t].
            image_seq: List[Optional[np.ndarray]] = []
            for _ in range(img_history_size):
                for k in camera_keys:
                    image_seq.append(curr["images"].get(k))

            # RDT outputs a horizon of actions; execute the first K steps like diffusion_policy eval.
            action_seq = model.step(curr["state"], image_seq, text_embeds)[0].cpu().numpy()  # (H,7)
            k_exec = max(1, min(int(args.steps_per_inference), int(action_seq.shape[0])))
            for k in range(k_exec):
                if quitter.quit_event.is_set():
                    print("Quit requested (q). Exiting control loop...")
                    raise KeyboardInterrupt
                # step_t0 = time.perf_counter()
                action7 = np.asarray(action_seq[k], dtype=np.float64)
                apply_action(robot_interface, args.controller_type, controller_cfg_obj, action7)
                # step_dt = time.perf_counter() - step_t0
                # if step_dt < target_dt:
                #     time.sleep(target_dt - step_dt)

            step_idx += 1

    finally:
        # Stop async workers first to avoid writing after resources are torn down.
        try:
            image_saver.stop(drain=True, timeout=5.0)
            print(f"Image saver stats: {image_saver.stats()}")
        except Exception:
            pass
        close_cameras(cam_by_key)
        try:
            robot_interface.close()
        except Exception:
            pass
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
