#!/usr/bin/env python3
"""
Run prompt-free SAM2 segmentation on a single image and write the binary mask.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CHECKPOINT = "./checkpoints/checkpoint_150.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but no GPU is available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_mask(image_path: Path, device: torch.device) -> np.ndarray:
    autocast_ctx = nullcontext()

    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else 0
        torch.cuda.set_device(dev_idx)
        if torch.cuda.get_device_properties(dev_idx).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    with autocast_ctx:
        sam2_model = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
        image_predictor = SAM2ImagePredictor(sam2_model)

        pil_image = Image.open(image_path).convert("RGB")
        image_predictor.set_image(pil_image)

        sparse_embeddings, dense_embeddings = image_predictor.model.sam_prompt_encoder(
            points=None, boxes=None, masks=None
        )
        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in image_predictor._features["high_res_feats"]
        ]
        low_res_masks, _, _, _ = image_predictor.model.sam_mask_decoder(
            image_embeddings=image_predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=image_predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        mask_logits = image_predictor._transforms.postprocess_masks(
            low_res_masks, image_predictor._orig_hw[-1]
        )
        prd_mask = torch.sigmoid(mask_logits[:, 0])

    cleaned = prd_mask.detach().cpu().squeeze()
    max_val = float(cleaned.max().item()) if cleaned.numel() else 1.0
    if max_val <= 0:
        max_val = 1.0
    normalized = cleaned / max_val
    binary_mask = (normalized >= 0.5).to(torch.uint8)

    return binary_mask.numpy() * 255


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SAM2 mask for a single image.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to save the mask image (default: <image>_sam_mask.png).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use, e.g. 'cuda:0' or 'cpu'. Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path = args.output
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_sam_mask.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    mask = _predict_mask(image_path, device)

    Image.fromarray(mask, mode="L").save(output_path)
    print(f"Mask saved to {output_path}")


if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/test/oxe-aug/sam2/sam2/inference_image.py \
/home/guanhuaji/test/oxe-aug/videos/example/0/1.png \
-o /home/guanhuaji/test/oxe-aug/videos/example/0/1_sam_mask.png
'''
