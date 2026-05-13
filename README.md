# 旧アニメ 4:3 → 16:9 / 4K 変換 実験プロジェクト

CEOからの依頼に基づく実験フレームワーク。Mac M系チップ（MPS）で動作する前提。

## ゴール
- 旧アニメ（4:3 SD画質）を **16:9 4K** に変換する独自パイプラインを試作
- 複数の手法を **比較検討** できる評価環境を整備
- パブリックドメイン素材で実験

## 構成
```
anime_upscale_experiment/
├── README.md                    ← このファイル
├── docs/
│   ├── 01_experiment_plan.md    ← 実験計画書（CEO報告用）
│   ├── 02_approach_comparison.md ← 4手法の比較分析
│   └── 03_results_template.md   ← 結果記録テンプレ
├── scripts/
│   ├── setup_env.sh             ← Mac M系セットアップ
│   ├── download_sample.py       ← パブリックドメイン素材取得
│   ├── pipeline.py              ← メインパイプライン
│   ├── upscale_module.py        ← 4K化モジュール（Real-ESRGAN）
│   ├── outpaint_module.py       ← 16:9拡張モジュール（SD-inpaint）
│   ├── crop_module.py           ← クロップ手法（比較用ベースライン）
│   ├── pillarbox_module.py      ← ピラーボックス手法
│   ├── temporal_check.py        ← 時間方向の整合性チェック
│   └── compare.py               ← サイドバイサイド比較生成
├── samples/                     ← 入力素材を配置
└── outputs/                     ← 変換結果
```

## クイックスタート
```bash
cd scripts
./setup_env.sh                   # 環境構築（初回のみ）
python download_sample.py        # パブリックドメイン素材ダウンロード
python pipeline.py --method all  # 4手法すべて実行
python compare.py                # 比較動画生成
```

## 設計判断のポイント
詳細は `docs/01_experiment_plan.md` 参照。要点：

1. **完全スクラッチ学習はしない** — Mac M系での生成モデル学習は非現実的。既存モデルの組み合わせ＋必要に応じてLoRA微調整で「カスタム」を実現。
2. **シーン適応的な手法選択** — 全フレーム同じ手法では破綻する。本プロジェクトの肝はここ。
3. **比較可能性を最優先** — 各手法の出力を必ず同フォーマットで保存し、定量・定性評価を可能に。
