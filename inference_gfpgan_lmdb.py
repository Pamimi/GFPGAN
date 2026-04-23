#!/usr/bin/env python3
"""对 LMDB 中的图像批量运行 GFPGAN；默认只处理前若干张。

与 inference_gfpgan.py 共用同一套模型加载与 enhance 逻辑。LMDB 内为已对齐人脸时，
使用 has_aligned=True，跳过整图人脸检测与 align/crop 流程（见 gfpgan.utils.GFPGANer.enhance）。
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.utils import imwrite

# 仓库根目录，用于导入 LMDBEngine
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.libs.utils_lmdb import LMDBEngine

from gfpgan import GFPGANer


def _tensor_chw_rgb_to_bgr_uint8(img_t: torch.Tensor) -> np.ndarray:
    """LMDB decode 为 RGB uint8 CHW -> OpenCV BGR HWC uint8."""
    if img_t.dim() != 3:
        raise ValueError(f"expect CHW image tensor, got shape {tuple(img_t.shape)}")
    rgb = img_t.detach().cpu()
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0, 255).to(torch.uint8)
    rgb_hwc = rgb.permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="GFPGAN inference on LMDB images (aligned faces, no face crop/det).")
    parser.add_argument("lmdb_path", type=str, help="LMDB 目录路径")
    parser.add_argument("-o", "--output", type=str, default="results_lmdb", help="输出目录")
    parser.add_argument(
        "-v", "--version", type=str, default="1.3", help="GFPGAN 版本: 1 | 1.2 | 1.3 | 1.4 | RestoreFormer"
    )
    parser.add_argument("-s", "--upscale", type=int, default=2, help="上采样倍数")
    parser.add_argument("-w", "--weight", type=float, default=0.5, help="restore 混合权重")
    parser.add_argument("-n", "--max-images", type=int, default=10, help="最多推理张数（默认 10）")
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="输出扩展名: png | jpg",
    )
    parser.add_argument(
        "--save-restored-only",
        action="store_true",
        help="仅保存单张还原图（restored_imgs），不写左右对比（默认会写对比到 cmp/）",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "restored_imgs"), exist_ok=True)
    if not args.save_restored_only:
        os.makedirs(os.path.join(args.output, "cmp"), exist_ok=True)

    # 与 inference_gfpgan.py 一致的版本与权重路径（不做背景 RealESRGAN，加快试跑）
    if args.version == "1":
        arch = "original"
        channel_multiplier = 1
        model_name = "GFPGANv1"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth"
    elif args.version == "1.2":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANCleanv1-NoCE-C2"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth"
    elif args.version == "1.3":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.3"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    elif args.version == "1.4":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.4"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif args.version == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        model_name = "RestoreFormer"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    else:
        raise ValueError(f"Wrong model version {args.version}.")

    model_path = os.path.join("experiments/pretrained_models", model_name + ".pth")
    if not os.path.isfile(model_path):
        model_path = os.path.join("gfpgan/weights", model_name + ".pth")
    if not os.path.isfile(model_path):
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None,
    )

    engine = LMDBEngine(args.lmdb_path, write=False)
    try:
        keys = engine.keys()[: args.max_images]
        for key in keys:
            print(f"Processing {key} ...")
            img_t = engine[key]
            input_bgr = _tensor_chw_rgb_to_bgr_uint8(img_t)

            cropped_faces, restored_faces, _restored_img = restorer.enhance(
                input_bgr,
                has_aligned=True,
                only_center_face=False,
                paste_back=True,
                weight=args.weight,
            )

            if not restored_faces:
                print(f"\tSkip {key}: no restored output.")
                continue

            safe_key = key.replace("/", "_")
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                out_name = f"{safe_key}_{idx:02d}.{args.ext}"
                if not args.save_restored_only:
                    # 与 inference_gfpgan.py 一致：左为送入网络的图（has_aligned 下为 resize 到 512 的输入），右为还原结果
                    cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                    imwrite(cmp_img, os.path.join(args.output, "cmp", out_name))
                imwrite(restored_face, os.path.join(args.output, "restored_imgs", out_name))
    finally:
        engine.close()

    if args.save_restored_only:
        print(f"Done. Restored images under [{os.path.join(args.output, 'restored_imgs')}]")
    else:
        print(
            f"Done. Comparison (原图|还原) under [{os.path.join(args.output, 'cmp')}], "
            f"single restored under [{os.path.join(args.output, 'restored_imgs')}]"
        )


if __name__ == "__main__":
    main()
