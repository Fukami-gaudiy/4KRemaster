"""手法A: Pillarbox - 左右に黒帯／ぼかし背景を入れて 16:9 化。

原画を一切変えないので最も安全。視聴体験は妥協。
左右をブラーした拡大版で埋める「ブラー・ピラーボックス」も実装。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def pillarbox(
    img: Image.Image,
    mode: str = "blur",
    blur_radius: int = 60,
    darken: float = 0.55,
) -> Image.Image:
    """4:3 画像の左右を埋めて 16:9 にする。

    Args:
        img: 入力画像。
        mode: "black" → 単純黒帯, "blur" → ぼかし背景。
        blur_radius: blur モード時のガウスぼかし強度。
        darken: blur モード時の明度低下係数 (0.0-1.0)。中央のコンテンツを浮き立たせる。

    Returns:
        16:9 画像。
    """
    w, h = img.size
    target_w = int(h * 16 / 9)
    if w >= target_w:
        return img  # 既に 16:9 以上

    canvas = Image.new("RGB", (target_w, h), (0, 0, 0))

    if mode == "blur":
        # 横幅を target_w まで拡大して中央クロップ → 強いブラー → 暗くする
        scale = target_w / w
        bg = img.resize((target_w, int(h * scale)), Image.LANCZOS)
        bg_top = (bg.size[1] - h) // 2
        bg = bg.crop((0, bg_top, target_w, bg_top + h))
        bg = bg.filter(ImageFilter.GaussianBlur(blur_radius))
        if darken < 1.0:
            arr = np.asarray(bg).astype(np.float32) * darken
            bg = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        canvas.paste(bg, (0, 0))

    # 中央に元画像
    offset = (target_w - w) // 2
    canvas.paste(img, (offset, 0))
    return canvas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--mode", choices=["black", "blur"], default="blur")
    args = parser.parse_args()

    src = Image.open(args.input).convert("RGB")
    dst = pillarbox(src, mode=args.mode)
    dst.save(args.output)
    print(f"Pillarboxed ({args.mode}): {src.size} -> {dst.size}")
