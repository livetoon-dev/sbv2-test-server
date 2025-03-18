style-bert-vits2を使った音声合成サーバー

1. システム構成：
```
├── app.py              # FastAPIサーバーのメインファイル
├── inference.py        # 音声合成の推論エンジン
├── models_config.json  # モデル設定ファイル
├── schema.py          # データスキーマ定義
└── scripts/
    └── tts_client.py  # クライアントライブラリ
```

2. セットアップ手順：
```bash
# 環境構築
uv sync

# サーバー起動
uv run app.py
```

3. 主な機能：

a) サーバー側（`app.py`）：
- エンドポイント：
  - `/tts`: 音声合成リクエストを受け付け
  - `/tts/{audio_id}`: 生成された音声データを取得
  - `/models`: 利用可能なモデル一覧を取得
  - `/models/switch`: 使用するモデルを切り替え

b) クライアント側（`tts_client.py`）：
```python
from scripts.tts_client import TTSClient

# クライアントの初期化
client = TTSClient(base_url="http://localhost:8000")

# 音声合成の実行
result = client.generate_speech(
    text="こんにちは、テストです。",
    model_id="uranai-emotion2",  # デフォルトモデル
    speaker_id="0",              # デフォルト話者
    output_dir="/path/to/output" # 出力ディレクトリ
)

if result:
    filepath, processing_time = result
    print(f"生成された音声ファイル: {filepath}")
    print(f"処理時間: {processing_time}秒")
```

4. モデル設定（`models_config.json`）：
```json
{
    "models": [
        {
            "model_id": "uranai-emotion2",
            "repo_id": "LiveTaro/uranai-emotion2",
            "name": "占い師感情2",
            "description": "感情表現が豊かな占い師の音声モデル",
            "speakers": {
                "0": "通常"
            },
            "is_default": true
        }
    ]
}
```

5. 主な特徴：

a) 自動デバイス選択：
- GPUが利用可能な場合は自動的にCUDAを使用
- GPU未対応環境ではCPUにフォールバック

b) 音声ファイル管理：
- タイムスタンプベースのファイル名生成
- 指定ディレクトリへの自動保存
- 処理時間の計測と詳細なログ出力

c) エラーハンドリング：
- モデル未存在エラー
- 話者ID無効エラー
- サーバー接続エラー

6. 使用例：

```python
# サーバーサイド（app.py実行）
uv run app.py

# クライアントサイド
from scripts.tts_client import TTSClient

client = TTSClient()

# 基本的な音声合成
result = client.generate_speech(
    text="こんにちは、音声合成のテストです。",
    output_dir="./outputs"
)

# 特定のモデルと話者を指定
result = client.generate_speech(
    text="感情豊かな音声を生成します。",
    model_id="uranai-emotion2",
    speaker_id="0",
    output_dir="./outputs"
)
```

7. 注意点：

- 環境変数の設定が必要：
  - `HUGGING_FACE_TOKEN`: Hugging Faceのアクセストークン
  - `MODELS_CACHE_DIR`: モデルのキャッシュディレクトリ（オプション）

- モデルの初回ダウンロード：
  - 初回実行時に指定されたモデルが自動的にダウンロードされます
  - ダウンロードには時間がかかる場合があります

- メモリ使用：
  - GPUメモリを使用する場合は十分なVRAMが必要です
  - CPUモードでも相応のメモリが必要です

このシステムは、style-bert-vits2を使用して高品質な音声合成を提供し、WebAPIとして簡単に利用できる設計になっています。
