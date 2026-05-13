"""手法B: Crop（上下切り）- ベースライン手法。

4:3 (1.333) を 16:9 (1.778) にするため、横を維持して上下を均等にクロップする。
構図が崩れるリスクは高いが、生成系手法と比較するベースラインとして必須。
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def crop_to_16_9(img: Image.Image, vertical_bias: float = 0.0) -> Image.Image:
    """4:3 画像を 16:9 にクロップ。

    Args:
        img: 入力画像（4:3 想定だが任意比でOK）
        vertical_bias: -1.0 (上寄せ) ～ +1.0 (下寄せ)。
                       人物の頭が切れることを避けたい場合は負の値を使う。
                       0.0 で中央クロップ。

    Returns:
        16:9 にクロップされた画像。
    """
    w, h = img.size
    target_h = int(w * 9 / 16)
    if target_h >= h:
        # 既に 16:9 以上に横長 → 横をクロップ
        target_w = int(h * 16 / 9)
        left = (w - target_w) // 2
        return img.crop((left, 0, left + target_w, h))

    excess = h - target_h
    # bias=0 で中央。bias=-1 で完全上寄せ、bias=+1 で完全下寄せ
    top = int(excess * (0.5 + vertical_bias * 0.5))
    top = max(0, min(excess, top))
    return img.crop((0, top, w, top + target_h))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="入力画像")
    parser.add_argument("output", help="出力画像")
    parser.add_argument("--bias", type=float, default=0.0)
    args = parser.parse_args()

    src = Image.open(args.input).convert("RGB")
    dst = crop_to_16_9(src, vertical_bias=args.bias)
    dst.save(args.output)
    print(f"Cropped: {src.size} -> {dst.size}")
