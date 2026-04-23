#!/usr/bin/env python3
"""对 LMDB 中的图像批量运行 GFPGAN；默认只处理前若干张。

与 inference_gfpgan.py 共用同一套模型加载与 enhance 逻辑。LMDB 内为已对齐人脸时，
使用 has_aligned=True，跳过整图人脸检测与 align/crop 流程（见 gfpgan.utils.GFPGANer.enhance）。
"""
#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import io
import os
import random
from warnings import warn

import lmdb
import torch
import torchvision
import numpy as np

class LMDBEngine:
    def __init__(self, lmdb_path, write=False):
        self._write = write
        self._manual_close = False
        self._lmdb_path = lmdb_path
        if write and not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        if write:
            self._lmdb_env = lmdb.open(
                lmdb_path, map_size=1099511627776
            )
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        else:
            self._lmdb_env = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=True
            ) 
            self._lmdb_txn = self._lmdb_env.begin(write=False)
        # print('Load lmdb length:{}.'.format(len(self.keys())))

    def __getitem__(self, key_name):
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError('Key:{} Not Found!'.format(key_name))
        try:
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            data = torchvision.io.decode_image(image_buf, mode=torchvision.io.ImageReadMode.RGB)
        except:
            data = torch.load(io.BytesIO(payload), weights_only=True)
        return data

    def __del__(self,):
        if not self._manual_close:
            warn('Writing engine not mannuly closed!', RuntimeWarning)
            self.close()

    def load(self, key_name, type='image', **kwargs):
        assert type in ['image', 'torch']
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError('Key:{} Not Found!'.format(key_name))
        if type == 'torch':
            torch_data = torch.load(io.BytesIO(payload), weights_only=True)
            return torch_data
        elif type == 'image':
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            if 'mode' in kwargs.keys():
                if kwargs['mode'].lower() == 'rgb':
                    _mode = torchvision.io.ImageReadMode.RGB
                elif kwargs['mode'].lower() == 'rgba':
                    _mode = torchvision.io.ImageReadMode.RGB_ALPHA
                elif kwargs['mode'].lower() == 'gray':
                    _mode = torchvision.io.ImageReadMode.GRAY
                elif kwargs['mode'].lower() == 'graya':
                    _mode = torchvision.io.ImageReadMode.GRAY_ALPHA
                else:
                    raise NotImplementedError
            else:
                _mode = torchvision.io.ImageReadMode.RGB
            image_data = torchvision.io.decode_image(image_buf, mode=_mode)
            return image_data
        else:
            raise NotImplementedError

    def dump(self, key_name, payload, type='image', encode_jpeg=True):
        assert isinstance(payload, torch.Tensor) or isinstance(payload, dict), payload
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not hasattr(self, '_dump_counter'):
            self._dump_counter = 0
        assert type in ['image', 'torch']
        if self._lmdb_txn.get(key_name.encode()):
            print('Key:{} exsists!'.format(key_name))
            return 
        if type == 'torch':
            assert isinstance(payload, torch.Tensor) or isinstance(payload, dict), payload
            torch_buf = io.BytesIO()
            if isinstance(payload, torch.Tensor):
                torch.save(payload.detach().float().cpu(), torch_buf)
            else:
                for key in payload.keys():
                    payload[key] = payload[key].detach().float().cpu()
                torch.save(payload, torch_buf)
            payload_encoded = torch_buf.getvalue()
            # torch_data = torch.load(io.BytesIO(payload_encoded), weights_only=True)
            self._lmdb_txn.put(key_name.encode(), payload_encoded)
        elif type == 'image':
            assert payload.dim() == 3 and payload.shape[0] == 3
            if payload.max() < 2.0:
                print('Image Payload Should be [0, 255].')
            if encode_jpeg:
                payload_encoded = torchvision.io.encode_jpeg(payload.to(torch.uint8), quality=95)
            else:
                payload_encoded = torchvision.io.encode_png(payload.to(torch.uint8))
            payload_encoded = b''.join(map(lambda x:int.to_bytes(x,1,'little'), payload_encoded.numpy().tolist()))
            self._lmdb_txn.put(key_name.encode(), payload_encoded)
        else:
            raise NotImplementedError
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def exists(self, key_name):
        if self._lmdb_txn.get(key_name.encode()):
            return True
        else:
            return False

    def delete(self, key_name):
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not self.exists(key_name):
            print('Key:{} Not Found!'.format(key_name))
            return
        deleted = self._lmdb_txn.delete(key_name.encode())
        if not deleted:
            print('Delete Failed: {}!'.format(key_name))
            return
        self._lmdb_txn.commit()
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def raw_load(self, key_name, ):
        raw_payload = self._lmdb_txn.get(key_name.encode())
        return raw_payload

    def raw_dump(self, key_name, raw_payload):
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not hasattr(self, '_dump_counter'):
            self._dump_counter = 0
        self._lmdb_txn.put(key_name.encode(), raw_payload)
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def keys(self, ):
        all_keys = list(self._lmdb_txn.cursor().iternext(values=False))
        all_keys = [key.decode() for key in all_keys]
        # print('Found data, length:{}.'.format(len(all_keys)))
        return all_keys

    def close(self, ):
        if self._write:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        self._lmdb_env.close()
        self._manual_close = True

    def random_visualize(self, vis_path, k=15, filter_key=None):
        all_keys = self.keys()
        if filter_key is not None:
            all_keys = [key for key in all_keys if filter_key in key]
        all_keys = random.choices(all_keys, k=k)
        print('Visualize: ', all_keys)
        images = [self.load(key, type='image')/255.0 for key in all_keys]
        images = [torchvision.transforms.functional.resize(i, (256, 256), antialias=True) for i in images]
        torchvision.utils.save_image(images, vis_path, nrow=5)


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

# from core.libs.utils_lmdb import LMDBEngine

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
