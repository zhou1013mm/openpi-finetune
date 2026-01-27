"""
Convert data from a run directory (e.g., run1/) into LeRobot v2 dataset format.

Expected files in data_dir (prefix can be any string, e.g., testing_demo):
  <prefix>_action.npz               (T, 8)
  <prefix>_camera_<camera_id>.npz   (T, H, W, 3)
  <prefix>_ee_states.npz            (T, 16)
  <prefix>_gripper_states.npz       (T,)
  <prefix>_joint_states.npz         (T, 7)
  <prefix>_task.npz                 (1, T) or (T,)
  config.json                       (contains task_name)

Usage:
  .venv/bin/python scripts/convert_run_dir_to_lerobot.py --data-dir /path/to/run1 --repo-id <org>/<name>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro


def _load_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "data" not in data:
        raise ValueError(f"Expected key 'data' in {path}")
    return data["data"]


def _find_prefixes(data_dir: Path) -> List[str]:
    prefixes = []
    for action_path in data_dir.glob("*_action.npz"):
        prefixes.append(action_path.name[: -len("_action.npz")])
    return sorted(set(prefixes))


def _get_task_name(data_dir: Path) -> str:
    config_path = data_dir / "config.json"
    if not config_path.exists():
        return "unknown"
    with config_path.open() as f:
        cfg = json.load(f)
    return str(cfg.get("task_name", "unknown"))


def _get_camera_files(data_dir: Path, prefix: str) -> Dict[str, Path]:
    camera_files: Dict[str, Path] = {}
    for cam_path in data_dir.glob(f"{prefix}_camera_*.npz"):
        cam_name = cam_path.stem.replace(f"{prefix}_", "")
        camera_files[cam_name] = cam_path
    return camera_files


def _validate_lengths(arrays: Dict[str, np.ndarray]) -> int:
    lengths = {k: v.shape[0] for k, v in arrays.items() if v is not None}
    if not lengths:
        raise ValueError("No arrays provided for length validation")
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Mismatched trajectory lengths: {lengths}")
    return next(iter(unique_lengths))


def _resolve_run_dirs(data_dir: Path, run_start: int | None, run_end: int | None) -> List[Path]:
    if any(data_dir.glob("*_action.npz")):
        return [data_dir]

    run_dirs = []
    if run_start is not None and run_end is not None:
        for i in range(run_start, run_end + 1):
            run_dir = data_dir / f"run{i}"
            if run_dir.exists():
                run_dirs.append(run_dir)
        if not run_dirs:
            raise RuntimeError(f"No run directories found in range run{run_start}..run{run_end}")
        return run_dirs

    # If no range provided, include all run* directories.
    run_dirs = sorted([p for p in data_dir.glob("run*") if p.is_dir()])
    if not run_dirs:
        raise RuntimeError("No run directories found; specify --run-start/--run-end or point to a single run dir")
    return run_dirs


def main(
    data_dir: str,
    repo_id: str,
    *,
    fps: int = 10,
    push_to_hub: bool = False,
    overwrite: bool = False,
    run_start: int | None = None,
    run_end: int | None = None,
    droid_format: bool = True,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise RuntimeError(f"Output path already exists: {output_path}")
        for child in output_path.iterdir():
            if child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
            else:
                child.unlink()
        output_path.rmdir()

    run_dirs = _resolve_run_dirs(data_dir, run_start, run_end)

    # Infer image shape from first run
    prefixes = _find_prefixes(run_dirs[0])
    if not prefixes:
        raise RuntimeError("No *_action.npz files found in first run; cannot infer prefixes")

    first_prefix = prefixes[0]
    camera_files = _get_camera_files(run_dirs[0], first_prefix)
    if not camera_files:
        raise RuntimeError("No camera files found matching *_camera_*.npz")
    sample_cam = next(iter(camera_files.values()))
    sample_cam_data = _load_npz(sample_cam)
    if sample_cam_data.ndim != 4:
        raise ValueError(f"Expected camera data shape (T, H, W, C), got {sample_cam_data.shape}")
    _, height, width, channels = sample_cam_data.shape

    if droid_format:
        features = {
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
            "prompt": {
                "dtype": "string",
                "shape": (1,),
                "names": ["prompt"],
            },
        }
    else:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "observation.ee_state": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["ee_state"],
            },
            "observation.gripper_state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        }
        for cam_name in camera_files.keys():
            features[f"observation.images.{cam_name}"] = {
                "dtype": "image",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="custom",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for run_dir in run_dirs:
        prefixes = _find_prefixes(run_dir)
        if not prefixes:
            raise RuntimeError(f"No *_action.npz files found in {run_dir}")

        task_name = _get_task_name(run_dir)
        for prefix in prefixes:
            action = _load_npz(run_dir / f"{prefix}_action.npz").astype(np.float32)
            joint = _load_npz(run_dir / f"{prefix}_joint_states.npz").astype(np.float32)
            ee = _load_npz(run_dir / f"{prefix}_ee_states.npz").astype(np.float32)
            gripper = _load_npz(run_dir / f"{prefix}_gripper_states.npz").astype(np.float32)
            if gripper.ndim == 1:
                gripper = gripper[:, None]

            cams = {name: _load_npz(path) for name, path in _get_camera_files(run_dir, prefix).items()}
            if not cams:
                raise RuntimeError(f"No camera files found in {run_dir} for prefix {prefix}")

            _validate_lengths(
                {
                    "action": action,
                    "joint": joint,
                    "ee": ee,
                    "gripper": gripper,
                    **{f"cam_{k}": v for k, v in cams.items()},
                }
            )

            num_frames = action.shape[0]
            cam_data = next(iter(cams.values()))
            for i in range(num_frames):
                if droid_format:
                    frame = {
                        "exterior_image_1_left": cam_data[i],
                        "exterior_image_2_left": cam_data[i],
                        "wrist_image_left": cam_data[i],
                        "joint_position": joint[i],
                        "gripper_position": gripper[i],
                        "actions": action[i],
                        "task": task_name,
                        "prompt": task_name,
                    }
                else:
                    frame = {
                        "observation.state": joint[i],
                        "observation.ee_state": ee[i],
                        "observation.gripper_state": gripper[i],
                        "action": action[i],
                        "task": task_name,
                    }
                    for cam_name, cam_arr in cams.items():
                        frame[f"observation.images.{cam_name}"] = cam_arr[i]
                dataset.add_frame(frame)

            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)