import time
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import runpy

import imageio.v3 as iio
import cv2
from PIL import Image
import argparse

MODEL_CHECKPOINT = "./checkpoints/checkpoint_150.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

REPO_ROOT = Path(__file__).resolve().parents[2]
ROVI_EXTENSION_DIR = REPO_ROOT / "rovi-aug-extension"
ROVI_CONFIG_PATH = ROVI_EXTENSION_DIR / "config.py"
_DATASET_CONFIG = None

# -------------------- Config helpers --------------------

def _load_dataset_config():
    """Load and cache the dataset config dictionary from rovi-aug-extension/config.py."""
    global _DATASET_CONFIG
    if _DATASET_CONFIG is not None:
        return _DATASET_CONFIG

    if not ROVI_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Unable to locate config.py at {ROVI_CONFIG_PATH}")

    rovi_root_str = str(ROVI_EXTENSION_DIR)
    if rovi_root_str not in sys.path:
        sys.path.insert(0, rovi_root_str)

    module_globals = runpy.run_path(str(ROVI_CONFIG_PATH))
    config_dict = module_globals.get("config")
    if not isinstance(config_dict, dict):
        raise ValueError(f"No 'config' dictionary found in {ROVI_CONFIG_PATH}")

    _DATASET_CONFIG = config_dict
    return _DATASET_CONFIG


def resolve_dataset_directory(dataset_name: str, split: str) -> Path:
    """Resolve dataset directory from config entry and split name."""
    config_dict = _load_dataset_config()
    dataset_cfg = config_dict.get(dataset_name)
    if dataset_cfg is None:
        sample_keys = ", ".join(sorted(config_dict.keys())[:8])
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config.py. Example keys: {sample_keys}"
        )

    out_path = dataset_cfg.get("out_path")
    if not out_path:
        raise ValueError(f"Dataset '{dataset_name}' is missing the 'out_path' field in config.py")

    dataset_root = Path(out_path)
    if not dataset_root.is_absolute():
        dataset_root = (ROVI_CONFIG_PATH.parent / dataset_root).resolve()

    resolved_dir = dataset_root / dataset_name / split
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Resolved dataset directory does not exist: {resolved_dir}")

    return resolved_dir

# -------------------- Utilities --------------------

def _remove_jpgs(directory: Path) -> int:
    removed = 0
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for img in directory.glob(pattern):
            img.unlink()
            removed += 1
    return removed


def extract_frames(video_path, frames_path):
    os.makedirs(frames_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = os.path.join(frames_path, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame_bgr)
    cap.release()


def to_hwc3_u8(arr) -> np.ndarray:
    """
    Normalize any frame-like array into shape (H, W, 3) uint8, contiguous.
    Handles:
      - 2D (H, W): replicate into 3 channels
      - (H, W, 1): replicate to 3
      - (3, H, W): transpose to (H, W, 3)
      - (H, W, 4): drop alpha
      - Other singleton/channel edge cases
    """
    a = np.asarray(arr)

    # Channel-first RGB -> HWC
    if a.ndim == 3 and a.shape[0] == 3 and (a.shape[1] > 3 and a.shape[2] > 3):
        a = np.transpose(a, (1, 2, 0))

    # Grayscale / channel count fixes
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        elif a.shape[-1] >= 3:
            a = a[..., :3]
        else:
            reps = 3 // max(1, a.shape[-1]) + (3 % max(1, a.shape[-1]) != 0)
            a = np.tile(a, (1, 1, reps))[..., :3]
    else:
        raise ValueError(f"Unsupported array shape for frame: {a.shape}")

    # dtype -> uint8
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(a, 0, 255)
            if a.max() <= 1.0:
                a = a * 255.0
        elif np.issubdtype(a.dtype, np.integer):
            a = np.clip(a, 0, 255)
        a = a.astype(np.uint8, copy=False)

    return np.ascontiguousarray(a)


def _mimwrite_auto(path, frames, fps: float):
    """
    Backend/codec-agnostic writer: stacks frames to (T,H,W,3) uint8 and lets imageio pick.
    """
    # Normalize and stack once; imageio treats first dimension as time axis
    stack = np.stack([to_hwc3_u8(f) for f in frames], axis=0)  # (T, H, W, 3), uint8
    iio.imwrite(path, stack, fps=float(fps))  # no plugin, no codec specified


def best_image_to_video_validation(video_path, mask_vidname, overlay_vidname,
                                   vid_predictor, img_predictor):

    # ---- Load frames (RGB) & fps ----
    cap = cv2.VideoCapture(video_path)
    frames_rgb = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    cap.release()

    if not frames_rgb:
        raise ValueError(f"No frames decoded from {video_path}")

    num_frames = len(frames_rgb)
    print(f"Loaded {num_frames} frames from {video_path}")

    # ---- SAM2 inference prep ----
    inference_state = vid_predictor.init_state(video_path=str(video_path))

    # pick an annotation frame by motion heuristic
    weights = [0.5 + 0 * abs(i - (num_frames / 2)) / (num_frames / 2) for i in range(num_frames)]
    frame_diffs = []
    for i in range(1, num_frames):
        prev_frame = frames_rgb[i - 1].astype(np.float32)
        curr_frame = frames_rgb[i].astype(np.float32)
        diff = np.sum(np.abs(curr_frame - prev_frame) > 50)
        weighted_diff = diff * weights[i]
        frame_diffs.append(weighted_diff)
    ann_frame_idx = int(np.argmax(frame_diffs) + 1)

    # one-shot image predictor pass
    img_predictor.set_image(Image.fromarray(frames_rgb[ann_frame_idx]))
    sparse_embeddings, dense_embeddings = img_predictor.model.sam_prompt_encoder(
        points=None, boxes=None, masks=None
    )

    batched_mode = False
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in img_predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = img_predictor.model.sam_mask_decoder(
        image_embeddings=img_predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=img_predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=batched_mode,
        high_res_features=high_res_features,
    )
    prd_masks = img_predictor._transforms.postprocess_masks(
        low_res_masks, img_predictor._orig_hw[-1]
    )

    # torch imported inside worker_fn; grab from globals
    torch = globals()["torch"]
    prd_mask = torch.sigmoid(prd_masks[:, 0])

    cleaned_mask = prd_mask.detach().cpu().squeeze()
    # Normalize to 0..1 then binarize at 0.5
    maxv = float(cleaned_mask.max().item()) if cleaned_mask.numel() else 1.0
    if maxv <= 0:
        maxv = 1.0
    clean_squash = cleaned_mask / maxv
    cleaned_mask = (clean_squash >= 0.5).to(cleaned_mask.dtype)

    _, _, _ = vid_predictor.add_new_mask(inference_state, ann_frame_idx, 1, cleaned_mask)

    video_segments = {}
    mask_frames = {}
    overlay_frames = {}

    # backward pass
    for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(
        inference_state, start_frame_idx=ann_frame_idx, reverse=True
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        out_obj_id, out_mask = list(video_segments[out_frame_idx].items())[0]
        mask_bool = np.squeeze(out_mask).astype(bool)
        m = (mask_bool.astype(np.uint8) * 255)
        mask_frames[out_frame_idx] = m

        overlay = frames_rgb[out_frame_idx].copy()
        overlay_mask = mask_bool
        overlay[overlay_mask] = (
            0.5 * overlay[overlay_mask] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay_frames[out_frame_idx] = overlay

    # forward pass
    for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(
        inference_state, start_frame_idx=ann_frame_idx, reverse=False
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        out_obj_id, out_mask = list(video_segments[out_frame_idx].items())[0]
        mask_bool = np.squeeze(out_mask).astype(bool)
        m = (mask_bool.astype(np.uint8) * 255)
        mask_frames[out_frame_idx] = m

        overlay = frames_rgb[out_frame_idx].copy()
        overlay_mask = mask_bool
        overlay[overlay_mask] = (
            0.5 * overlay[overlay_mask] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay_frames[out_frame_idx] = overlay

    # Sort indices and normalize frames to HxWx3 u8
    ordered_indices = sorted(mask_frames.keys())
    mask_list = [to_hwc3_u8(mask_frames[idx]) for idx in ordered_indices]
    overlay_list = [to_hwc3_u8(overlay_frames[idx]) for idx in ordered_indices]

    # ---------- Write videos (backend/codec auto) ----------
    _mimwrite_auto(mask_vidname, mask_list, fps=float(fps))
    _mimwrite_auto(overlay_vidname, overlay_list, fps=float(fps))

    return {
        "video_segments": video_segments,
        "composites": overlay_list,
    }


def worker_fn(episodes, directory, gpu_idx, counter, mute_output):
    if mute_output:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    import torch
    globals()['torch'] = torch
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda")

    if device.type == "cuda":
        autocast_cm = torch.autocast("cuda", dtype=torch.bfloat16)
        autocast_cm.__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    vid_predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    sam2_model    = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    img_predictor = SAM2ImagePredictor(sam2_model)

    for ep in episodes:
        episode_start = time.perf_counter()
        ep_dir = Path(directory) / str(ep)
        video_path = ep_dir / "source_video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing source_video.mp4 in {ep_dir}")

        mask_mp4 = ep_dir / "sam_mask.mp4"
        overlay_mp4 = ep_dir / "sam_overlay.mp4"

        print(mask_mp4, overlay_mp4)

        best_image_to_video_validation(
            str(video_path), str(mask_mp4), str(overlay_mp4),
            vid_predictor, img_predictor
        )

        elapsed = time.perf_counter() - episode_start
        print(f"[TIMER] Episode {ep} completed in {elapsed:.2f}s")

        with counter.get_lock():
            counter.value += 1


# ────────────────────────── Main entry ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parallel call to worker_fn by episode range or auto discovery")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--start', type=int, default=None, help='Start episode (inclusive)')
    parser.add_argument('--end',   type=int, default=None, help='End episode (exclusive)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset key defined in rovi-aug-extension/config.py')
    parser.add_argument('--split', type=str, required=True,
                        help='Dataset split to process (e.g. train, val, test)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel processes / GPUs (≤ visible GPUs)')
    args = parser.parse_args()

    try:
        dataset_dir = resolve_dataset_directory(args.dataset, args.split)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))

    dataset_dir_str = str(dataset_dir)

    # — Generate list of episodes to process —
    if args.start is not None and args.end is not None:
        episodes = list(range(args.start, args.end))
    else:
        episodes = sorted(int(p.name) for p in dataset_dir.iterdir()
                          if p.is_dir() and p.name.isdigit())
    if not episodes:
        raise SystemExit("No episode directories found")

    total = len(episodes)
    chunks = [[] for _ in range(args.num_workers)]
    for idx, ep in enumerate(episodes):
        chunks[idx % args.num_workers].append(ep)

    # Shared counter
    counter = mp.Value('i', 0)

    # Launch child processes
    mute_output = not args.verbose
    procs = []
    for gpu_idx, ep_chunk in enumerate(chunks):
        if not ep_chunk:
            continue
        p = mp.Process(
            target=worker_fn,
            args=(ep_chunk, dataset_dir_str, gpu_idx, counter, mute_output),
            daemon=False
        )
        p.start()
        procs.append(p)

    # Single tqdm progress bar
    last = 0
    with tqdm(total=total, desc="Episodes progress", unit="ep") as pbar:
        while any(p.is_alive() for p in procs):
            time.sleep(1)  # refresh rate 1 s
            with counter.get_lock():
                done = counter.value
            if done > last:
                pbar.update(done - last)
                last = done
        # ensure completion update
        pbar.update(total - last)

    # Wait for child processes
    for p in procs:
        p.join()

    print("✅ All episodes processed")


if __name__ == "__main__":
    main()