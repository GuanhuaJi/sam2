import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "rovi-aug-extension"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from config import config
from typing import List, Tuple, Set

import cv2 as cv
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
def _phase_corr_shift(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, int]:
    f1, f2 = img1.astype(np.float32), img2.astype(np.float32)
    (dx, dy), _ = cv.phaseCorrelate(f1, f2)
    return int(round(dx)), int(round(dy))

def clip_to_smaller(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])

    def center_crop(img: np.ndarray, hh: int, ww: int) -> np.ndarray:
        y0 = (img.shape[0] - hh) // 2
        x0 = (img.shape[1] - ww) // 2
        return img[y0:y0 + hh, x0:x0 + ww]

    if a.shape[:2] != (h, w):
        a = center_crop(a, h, w)
    if b.shape[:2] != (h, w):
        b = center_crop(b, h, w)
    return a, b

def _brute_force_iou(sim: np.ndarray, sam: np.ndarray, r: int) -> Tuple[int, int]:
    sim_b, sam_b = sim > 0, sam > 0
    best_iou, best = -1.0, (0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            rolled = np.roll(sim_b, shift=(dy, dx), axis=(0, 1))
            if dy > 0:  rolled[:dy, :] = False
            elif dy < 0: rolled[dy:, :] = False
            if dx > 0:  rolled[:, :dx] = False
            elif dx < 0: rolled[:, dx:] = False
            inter = np.logical_and(rolled, sam_b).sum()
            union = np.logical_or (rolled, sam_b).sum()
            iou = inter / union if union else 0
            if iou > best_iou:
                best_iou, best = iou, (dx, dy)
    return best

def best_shift_fast(sim: np.ndarray, sam: np.ndarray,
                    max_pix: int = 10) -> Tuple[int, int]:
    if not (sim.any() and sam.any()):
        return 0, 0
    try:
        dx, dy = _phase_corr_shift(sim, sam)
        if abs(dx) > max_pix or abs(dy) > max_pix:
            raise ValueError
    except Exception:
        dx, dy = _brute_force_iou(sim, sam, max_pix)
    return dx, dy

def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
    if dy > 0:  shifted[:dy, :] = 0
    elif dy < 0: shifted[dy:, :] = 0
    if dx > 0:  shifted[:, :dx] = 0
    elif dx < 0: shifted[:, dx:] = 0
    return shifted

def discover_common_episodes(sim_root: Path, sam_root: Path) -> List[int]:
    def _numeric_dirs(root: Path) -> Set[int]:
        return {int(p.name) for p in root.iterdir()
                if p.is_dir() and p.name.isdigit()}
    eps = _numeric_dirs(sim_root) & _numeric_dirs(sam_root)
    return sorted(eps)

DEFAULT_OUTPUT_ROOT = Path('/home/guanhuaji/test/oxe-aug/videos')


def _resolve_roots(dataset: str, split: str):
    dataset_cfg = config.get(dataset, {})
    if not dataset_cfg:
        raise ValueError(f"Unknown dataset {dataset}")

    out_cfg = dataset_cfg.get('out_path')
    if out_cfg:
        out_root = Path(out_cfg)
        if not out_root.is_absolute():
            out_root = Path(DEFAULT_OUTPUT_ROOT) / out_root
    else:
        out_root = Path(DEFAULT_OUTPUT_ROOT)

    sim_root = out_root / dataset / split

    sam_root_cfg = dataset_cfg.get('sam_root')
    if sam_root_cfg:
        sam_root = Path(sam_root_cfg) / split
    else:
        sam_root = sim_root
    return sim_root, sam_root

def process_episode(args: Tuple[int, Path, Path, int, int, bool, bool, bool]) -> Tuple[int, bool, str]:
    episode, sim_root, sam_root, max_pix, sam_radius, sim_only, sam_only, add_unmoved = args
    sim_mp4 = sim_root / f"{episode}" / "mask.mp4"
    sam_mp4 = sam_root / str(episode) / "sam_mask.mp4"
    out_mp4 = sam_root / str(episode) / "merged_mask.mp4"

    if not sim_mp4.exists():
        return episode, False, f"mask video not found: {sim_mp4}"
    if not sam_mp4.exists():
        return episode, False, f"sam mask video not found: {sam_mp4}"

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cap_sim = cv.VideoCapture(str(sim_mp4))
    cap_sam = cv.VideoCapture(str(sam_mp4))
    if not cap_sim.isOpened():
        return episode, False, f"cannot open mask video: {sim_mp4}"
    if not cap_sam.isOpened():
        cap_sim.release()
        return episode, False, f"cannot open sam mask video: {sam_mp4}"

    sim_total = int(cap_sim.get(cv.CAP_PROP_FRAME_COUNT))
    sam_total = int(cap_sam.get(cv.CAP_PROP_FRAME_COUNT))
    if sim_total != sam_total:
        cap_sim.release(); cap_sam.release()
        return episode, False, f"length mismatch: sim={sim_total}, sam={sam_total}"

    frames_written = 0
    mask_frames = []
    try:
        while True:
            ok_sim, sim_frm = cap_sim.read()
            ok_sam, sam_frm = cap_sam.read()
            if not ok_sim or not ok_sam:
                break

            sim_g = sim_frm
            if sim_g.ndim == 3:
                sim_g = cv.cvtColor(sim_g, cv.COLOR_BGR2GRAY)
            sam_g = sam_frm
            if sam_g.ndim == 3:
                sam_g = cv.cvtColor(sam_g, cv.COLOR_BGR2GRAY)

            sim_g, sam_g = clip_to_smaller(sim_g, sam_g)

            dx, dy = best_shift_fast(sim_g, sam_g, max_pix=max_pix)
            sim_shift = shift_mask(sim_g, dx, dy)

            sam_b = sam_g > 125
            if sam_radius > 0:
                sim_b = (sim_shift > 125)
                k = 2 * sam_radius + 1
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
                near_sim = cv.dilate(sim_b.astype(np.uint8), kernel).astype(bool)
                sam_b &= near_sim

            if sam_only:
                union_b = sam_b
            elif sim_only:
                union_b = sim_shift > 125
            else:
                union_b = (sim_shift > 125) | sam_b

            if add_unmoved:
                union_b |= (sim_g > 125)

            mask_frame = (union_b.astype(np.uint8) * 255)
            mask_frame = np.stack([mask_frame] * 3, axis=-1)
            mask_frames.append(mask_frame)
            frames_written += 1
    finally:
        cap_sim.release()
        cap_sam.release()

    if mask_frames:
        stack = np.stack(mask_frames, axis=0).astype(np.uint8)
        iio.imwrite(str(out_mp4), stack, fps=30, codec='libx264')

    return episode, True, f"{frames_written} frames"

def parse_episode_list(spec: str) -> List[int]:
    eps: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            eps.extend(range(a, b + 1))
        else:
            eps.append(int(part))
    return sorted(set(eps))

def build_episode_range(start: int | None, end: int | None,
                        episodes_str: str | None) -> List[int]:
    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("Both --start and --end must be given together")
        if end < start:
            raise ValueError("--end must be ≥ --start")
        return list(range(start, end + 1))
    if not episodes_str:
        raise ValueError
    return parse_episode_list(episodes_str)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset", required=True, help="Dataset name")
    p.add_argument("--split", required=True, help="Dataset split (train/val/test)")
    p.add_argument("--start", type=int, default=None, help="first episode (inclusive)")
    p.add_argument("--end",   type=int, default=None, help="last  episode (exclusive)")
    p.add_argument("--num_workers", type=int, default=min(20, cpu_count()), help="number of processes")
    p.add_argument("--max_pix",    type=int, default=10, help="best-shift search radius (px)")
    p.add_argument("--sam_radius", type=int, default=60, help="sam mis-detection crop radius; 0 = disable")
    p.add_argument("--sim_only", action="store_true", help="use only sim mask, skip sam mask")
    p.add_argument("--sam_only", action="store_true", help="use only sam mask (after crop)")
    p.add_argument("--add_unmoved_sim", action="store_true",
                   help="re-add original sim mask")
    return p.parse_args()

def main():
    args = parse_args()

    if args.sim_only and args.sam_only:
        sys.exit("Cannot use --sim_only and --sam_only together")

    if args.start is None or args.end is None:
        sys.exit("--start and --end are required together")
    if args.end <= args.start:
        sys.exit("--end must be greater than --start")

    sim_root, sam_root = _resolve_roots(args.dataset, args.split)
    episodes = list(range(args.start, args.end))

    tasks = [(ep, sim_root, sam_root,
              args.max_pix, args.sam_radius,
              args.sim_only, args.sam_only, args.add_unmoved_sim) for ep in episodes]

    print(f"Total {len(tasks)} episodes {args.num_workers} processes")
    ok_cnt = fail_cnt = 0
    with get_context("spawn").Pool(args.num_workers) as pool:
        for ep, ok, msg in tqdm(pool.imap_unordered(process_episode, tasks),
                                total=len(tasks), unit="ep"):
            if ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                print(f"⚠ Episode {ep}: {msg}", file=sys.stderr)
    print(f"✓ Done. success={ok_cnt}, failed={fail_cnt}")

if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/test/oxe-aug/sam2/merge_mask.py --dataset jaco_play --split train --start 0 --end 5 --max_pix 10
'''