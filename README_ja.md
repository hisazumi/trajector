# Trajector 🎯

YOLOv8とOpenCVを使用したリアルタイム物体追跡・軌跡可視化システム

## 機能

- **物体検出・追跡**: YOLOv8による リアルタイム物体検出と永続的ID追跡
- **軌跡可視化**: カスタマイズ可能な軌跡エフェクトで物体の動きを表示
- **ヒートマップ生成**: 物体の存在と移動パターンを可視化するヒートマップ作成
- **複数の入力ソース**: ビデオファイルとWebカメラストリームをサポート
- **Webインターフェース**: Streamlitベースのインタラクティブなウェブアプリケーション
- **CLIツール**: バッチ処理用のコマンドラインインターフェース

## インストール

### 前提条件

- Python 3.12以上
- [uv](https://github.com/astral-sh/uv) (Pythonパッケージマネージャー)

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/hisazumi/trajector.git
cd trajector

# uvを使用して依存関係をインストール
uv sync

# YOLOv8モデルをダウンロード（初回実行時に自動ダウンロード）
# または手動でプロジェクトルートにダウンロード：
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

## 使用方法

### Webインターフェース (Streamlit)

インタラクティブなWebアプリケーションを起動：

```bash
uv run streamlit run src/web_app.py
```

機能：
- **ビデオアップロード**: 進捗表示付きでビデオファイルを処理
- **Webカメラトラッキング**: 録画機能付きのリアルタイム物体追跡
- **インタラクティブコントロール**: リアルタイムで追跡パラメータを調整
- **結果のダウンロード**: 処理済みビデオとヒートマップをエクスポート

### コマンドラインインターフェース

#### ビデオファイルの処理

```bash
# 基本的な使用方法
uv run python -m src.cli file input_video.mp4

# オプション付き
uv run python -m src.cli file input_video.mp4 \
  --output output_video.mp4 \
  --config config/config.yaml \
  --heatmap \
  --show-status
```

オプション：
- `-o, --output`: 出力ビデオファイルパス
- `-c, --config`: 設定ファイル (デフォルト: config/config.yaml)
- `--heatmap`: 軌跡ヒートマップを生成
- `--show-status`: ビデオにステータスオーバーレイを表示
- `--no-preview`: プレビューウィンドウを無効化
- `--quiet`: 進捗出力を抑制

#### Webカメラトラッキング

```bash
# Webカメラトラッキングを開始
uv run python -m src.cli webcam

# 録画付き
uv run python -m src.cli webcam --save --output output/
```

キーボードショートカット：
- `q`: 終了
- `s`: スクリーンショット保存
- `h`: ヒートマップ保存
- `r`: トラッキングをリセット
- `t`: 動的ヒートマップオーバーレイの切り替え
- `+/-`: ヒートマップの不透明度を増減

## 設定

`config/config.yaml`を編集してカスタマイズ：

```yaml
detector:
  model_path: "yolov8n.pt"  # YOLOv8モデルファイル
  device: "cpu"              # cpuまたはcuda
  confidence: 0.5            # 検出信頼度の閾値

tracker:
  max_disappeared: 30        # 物体を削除するまでのフレーム数
  max_distance: 50          # 物体関連付けの最大距離

visualizer:
  trajectory_length: 30      # 軌跡の点数
  trajectory_color: [0, 255, 0]  # 軌跡の色 (BGR)
  trajectory_thickness: 2    # 軌跡の線の太さ
  show_bbox: true           # バウンディングボックスを表示
  show_id: true             # 物体IDを表示
  show_trajectory: true     # 軌跡を表示
  show_heatmap: false       # 動的ヒートマップオーバーレイ
  heatmap_alpha: 0.6        # ヒートマップの透明度
```

## プロジェクト構造

```
trajector/
├── src/
│   ├── cli.py              # コマンドラインインターフェース
│   ├── web_app.py          # Streamlitウェブアプリケーション
│   ├── core/               # コア追跡コンポーネント
│   │   ├── detector.py     # YOLO検出器ラッパー
│   │   ├── tracker.py      # 物体追跡ロジック
│   │   └── visualizer.py   # 可視化ユーティリティ
│   ├── sources/            # 入力ソース
│   │   ├── video.py        # ビデオファイルハンドラ
│   │   └── webcam.py       # Webカメラハンドラ
│   └── processors/         # 処理パイプライン
│       └── pipeline.py     # メイン処理パイプライン
├── config/
│   └── config.yaml         # 設定ファイル
├── tests/                  # ユニットテストと統合テスト
├── examples/               # サンプルビデオと出力
└── output/                 # 出力ディレクトリ
```

## 使用例

### ビデオの処理

```bash
# デフォルト設定で処理
uv run python -m src.cli file examples/sample_people.mp4

# ヒートマップとステータスオーバーレイを生成
uv run python -m src.cli file examples/sample_people.mp4 \
  --heatmap --show-status
```

### Webインターフェースの起動

```bash
uv run streamlit run src/web_app.py
```

その後、ブラウザで http://localhost:8501 を開きます。

## テスト

テストスイートの実行：

```bash
# すべてのテストを実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=src --cov-report=html
```

## 必要なパッケージ

主な依存関係：
- `opencv-python`: コンピュータビジョン操作
- `ultralytics`: YOLOv8物体検出
- `numpy`: 数値計算
- `streamlit`: Webインターフェース
- `pyyaml`: 設定管理

完全な依存関係リストは`pyproject.toml`を参照してください。

## ライセンス

MITライセンス

## コントリビュート

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## 謝辞

- [YOLOv8](https://github.com/ultralytics/ultralytics) - 物体検出
- [OpenCV](https://opencv.org/) - コンピュータビジョン操作
- [Streamlit](https://streamlit.io/) - Webインターフェース