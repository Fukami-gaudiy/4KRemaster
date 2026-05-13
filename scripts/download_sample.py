#!/usr/bin/env python3
"""パブリックドメインのアニメサンプル素材を取得する。

著作権リスクを避けるため Internet Archive 上の確実にPDの素材のみ。
初期実験用に短いクリップ (1分以内) を推奨。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

# 確認済みパブリックドメイン素材（米国PD）
PD_SAMPLES = {
    "betty_boop_snow_white": {
        "title": "Betty Boop - Snow-White (1933)",
        "url": "https://archive.org/download/Snow-white1933/Snow-white1933.mp4",
        "aspect": "4:3",
        "duration_sec": 422,
        "notes": "Fleischer Studios. 米国PD。線画の質感が実験に好適。",
    },
    "superman_mad_scientist": {
        "title": "Superman: The Mad Scientist (1941)",
        "url": "https://archive.org/download/Superman-TheMadScientist/Superman-TheMadScientist.mp4",
        "aspect": "4:3",
        "duration_sec": 600,
        "notes": "Fleischer Superman. 米国PD。色彩豊か、構図ダイナミック。",
    },
    "popeye_aladdin": {
        "title": "Popeye - Aladdin and His Wonderful Lamp (1939)",
        "url": "https://archive.org/download/PopeyeMeetsAladdinAndHisWonderfulLamp1939/PopeyeMeetsAladdinAndHisWonderfulLamp1939.mp4",
        "aspect": "4:3",
        "duration_sec": 1234,
        "notes": "2-color テクニカラー。リスク高めなので使用前に再確認推奨。",
    },
}


def _progress(block: int, block_size: int, total: int) -> None:
    pct = min(100, block * block_size * 100 // max(total, 1))
    sys.stdout.write(f"\r  ダウンロード中... {pct}%")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="PDサンプル素材ダウンロード")
    parser.add_argument(
        "--id",
        choices=list(PD_SAMPLES.keys()) + ["all"],
        default="betty_boop_snow_white",
        help="ダウンロードする素材ID",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "samples",
        help="出力ディレクトリ",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ids = list(PD_SAMPLES.keys()) if args.id == "all" else [args.id]
    for sample_id in ids:
        sample = PD_SAMPLES[sample_id]
        dest = args.out / f"{sample_id}.mp4"
        print(f"[{sample_id}] {sample['title']}")
        print(f"  アスペクト: {sample['aspect']}, 約{sample['duration_sec']}秒")
        print(f"  備考: {sample['notes']}")
        if dest.exists():
            print(f"  既に存在: {dest} (スキップ)")
            continue
        try:
            urlretrieve(sample["url"], dest, _progress)
            print(f"\n  保存: {dest}")
        except Exception as exc:  # pragma: no cover
            print(f"\n  失敗: {exc}")
            print(f"  手動で {sample['url']} を取得して {dest} に配置してください")


if __name__ == "__main__":
    main()
