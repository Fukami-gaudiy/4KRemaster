"""メインパイプライン：動画 4:3 → 16:9 4K 変換。

使用例:
    python pipeline.py --input ../samples/betty_boop_snow_white.mp4 --method hybrid
    python pipeline.py --input INPUT --method all   # 4手法すべて実行（比較用）

処理:
    1. ffmpeg でフレーム分解 → frames/{method}/raw/000000.png ...
    2. シーン分類（hybrid のみ）
    3. アスペクト比変換（method に応じて pillarbox/crop/outpaint）
    4. 4K アップスケール (Real-ESRGAN)
    5. ffmpeg でフレーム再結合＋音声マージ → outputs/{method}/result.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from PIL import Image
from tqdm import tqdm

# 同階層モジュール
from crop_module import crop_to_16_9
from pillarbox_module import pillarbox

METHODS = ["pillarbox", "crop", "outpaint", "hybrid"]


@dataclass
class FrameMeta:
    index: int
    scene: str | None = None
    method_used: str | None = None


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_frames(video: Path, out_dir: Path, fps: int = 24) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "1",
        str(out_dir / "%06d.png"),
    ])
    return len(sorted(out_dir.glob("*.png")))


def combine_frames(frames_dir: Path, audio_src: Path, out_video: Path, fps: int = 24) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-i", str(audio_src),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_video),
    ])


def _aspect_convert(method: str, img: Image.Image, outpaint_fn: Callable | None) -> Image.Image:
    if method == "pillarbox":
        return pillarbox(img, mode="blur")
    if method == "crop":
        return crop_to_16_9(img, vertical_bias=-0.2)  # 頭切れを避けるため少し上寄せ
    if method == "outpaint":
        assert outpaint_fn is not None
        return outpaint_fn(img)
    raise ValueError(method)


def process(
    video: Path,
    method: str,
    work_root: Path,
    out_root: Path,
    fps: int = 24,
    do_upscale: bool = True,
    max_frames: int | None = None,
) -> None:
    assert method in METHODS

    raw_dir = work_root / method / "raw"
    converted_dir = work_root / method / "converted"
    upscaled_dir = work_root / method / "upscaled"
    for d in (raw_dir, converted_dir, upscaled_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{method}] フレーム分解 ===")
    if not any(raw_dir.glob("*.png")):
        extract_frames(video, raw_dir, fps=fps)

    frames = sorted(raw_dir.glob("*.png"))
    if max_frames is not None:
        frames = frames[:max_frames]

    # 遅延 import（重い依存を method に応じてのみロード）
    outpaint_fn = None
    classifier = None
    upscaler = None

    needs_outpaint = method in ("outpaint", "hybrid")
    if needs_outpaint:
        print("Outpaint パイプラインを初期化中…")
        from outpaint_module import outpaint  # noqa: WPS433
        outpaint_fn = outpaint

    if method == "hybrid":
        print("シーン分類器を初期化中…")
        from scene_classifier import SceneClassifier, recommended_method  # noqa: WPS433
        classifier = SceneClassifier()
        _recommend = recommended_method
    else:
        _recommend = None  # type: ignore

    if do_upscale:
        print("4K アップスケーラを初期化中…")
        from upscale_module import upscale_to_4k  # noqa: WPS433
        upscaler = upscale_to_4k

    meta_log: list[dict] = []
    print(f"=== [{method}] アスペクト変換 + 4K化 ({len(frames)} frames) ===")
    for f in tqdm(frames):
        img = Image.open(f).convert("RGB")

        # 1) アスペクト変換
        if method == "hybrid":
            assert classifier is not None and _recommend is not None
            res = classifier.classify(img)
            sub_method = _recommend(res.scene)
            converted = _aspect_convert(sub_method, img, outpaint_fn)
            meta_log.append(asdict(FrameMeta(int(f.stem), res.scene, sub_method)))
        else:
            converted = _aspect_convert(method, img, outpaint_fn)
            meta_log.append(asdict(FrameMeta(int(f.stem), None, method)))

        converted.save(converted_dir / f.name)

        # 2) 4K化
        if do_upscale and upscaler is not None:
            up = upscaler(converted)
            up.save(upscaled_dir / f.name)

    # メタログ
    with (work_root / method / "frames_meta.json").open("w") as fp:
        json.dump(meta_log, fp, indent=2)

    # 3) 動画再結合
    print(f"=== [{method}] 動画再結合 ===")
    src_dir = upscaled_dir if do_upscale else converted_dir
    out_video = out_root / method / "result.mp4"
    combine_frames(src_dir, audio_src=video, out_video=out_video, fps=fps)
    print(f"出力: {out_video}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="入力動画 (4:3 想定)")
    parser.add_argument("--method", choices=METHODS + ["all"], default="hybrid")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--no-upscale", action="store_true", help="アスペクト変換のみ実施（高速確認用）")
    parser.add_argument("--max-frames", type=int, default=None, help="最初の N フレームのみ処理（PoC用）")
    parser.add_argument("--work", type=Path, default=Path(__file__).parent.parent / "outputs" / "_work")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "outputs")
    args = parser.parse_args()

    methods = METHODS if args.method == "all" else [args.method]
    for m in methods:
        process(
            video=args.input,
            method=m,
            work_root=args.work,
            out_root=args.out,
            fps=args.fps,
            do_upscale=not args.no_upscale,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
