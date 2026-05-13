"""4K 化モジュール - Real-ESRGAN anime6B モデル。

旧アニメ・イラスト向けに学習されたモデル。線画のディテール再現が良好。
Mac M系の MPS でも動作する（FP32 推論、メモリ余裕があれば 1080p 入力までOK）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_TARGET_4K_W = 3840
_TARGET_4K_H = 2160


def upscale_to_4k(
    img: Image.Image,
    model_path: str | Path | None = None,
    half: bool = False,
) -> Image.Image:
    """画像を 4K (3840x2160) に近い解像度まで Real-ESRGAN で拡大する。

    入力アスペクト比を維持しつつ、長辺が 3840 に達するまで x4 アップスケール。
    必要なら最後に LANCZOS で 4K にスナップする。
    """
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    from realesrgan import RealESRGANer  # type: ignore

    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "RealESRGAN_x4plus_anime_6B.pth"
    model_path = str(model_path)

    # anime6B は 6 blocks の RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    device = _device()

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,         # Mac でメモリ抑制のためタイル処理
        tile_pad=10,
        pre_pad=0,
        half=half and device.type == "cuda",  # MPS は half 非対応の場合あり
        device=device,
    )

    arr = np.array(img.convert("RGB"))
    out, _ = upsampler.enhance(arr, outscale=4)
    up = Image.fromarray(out)

    # 4K に近づけるための最終調整
    w, h = up.size
    target_aspect = w / h
    if target_aspect >= 16 / 9:
        new_w = _TARGET_4K_W
        new_h = int(round(_TARGET_4K_W / target_aspect))
    else:
        new_h = _TARGET_4K_H
        new_w = int(round(_TARGET_4K_H * target_aspect))
    if (new_w, new_h) != up.size:
        up = up.resize((new_w, new_h), Image.LANCZOS)
    return up


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    src = Image.open(args.input).convert("RGB")
    dst = upscale_to_4k(src, model_path=args.model)
    dst.save(args.output)
    print(f"Upscaled: {src.size} -> {dst.size}")
