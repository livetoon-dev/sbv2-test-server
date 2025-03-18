import requests
import json
from typing import Optional
import io
from pathlib import Path
from datetime import datetime
import time
import os

class TTSClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def generate_speech(
        self,
        text: str,
        model_id: str = "uranai-emotion2",
        speaker_id: str = "0",
        output_dir: str = "/home/ubuntu/dev/outputs"
    ) -> Optional[tuple[str, float]]:
        """
        テキストから音声を生成します

        Args:
            text (str): 音声合成するテキスト
            model_id (str): 使用するモデルのID（デフォルト: "uranai-emotion2"）
            speaker_id (str): 話者ID（デフォルト: "0"）
            output_dir (str): 出力ディレクトリのパス

        Returns:
            tuple[str, float] | None: (生成された音声ファイルのパス, 処理時間（秒）) または None（エラー時）
        """
        start_time = time.time()
        print(f"[{datetime.now().isoformat()}] 音声合成を開始します")
        print(f"テキスト: {text}")
        print(f"モデルID: {model_id}")
        print(f"話者ID: {speaker_id}")

        # 出力ディレクトリの作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TTSリクエストを送信
        tts_url = f"{self.base_url}/tts"
        payload = {
            "text": text,
            "model_id": model_id,
            "speaker_id": speaker_id
        }

        try:
            print(f"[{datetime.now().isoformat()}] TTSサーバーにリクエストを送信中...")
            request_start_time = time.time()
            response = requests.post(tts_url, json=payload)
            response.raise_for_status()
            result = response.json()
            audio_id = result["id"]
            request_time = time.time() - request_start_time
            print(f"TTSリクエスト完了（{request_time:.2f}秒）")

            # 生成された音声データを取得
            print(f"[{datetime.now().isoformat()}] 音声データを取得中...")
            audio_start_time = time.time()
            audio_url = f"{self.base_url}/tts/{audio_id}"
            audio_response = requests.get(audio_url)
            audio_response.raise_for_status()
            audio_time = time.time() - audio_start_time
            print(f"音声データ取得完了（{audio_time:.2f}秒）")
            
            # ファイル名を生成（日時とテキストの先頭を含む）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_prefix = text[:20].replace(" ", "_").replace("/", "_")  # ファイル名に使えない文字を置換
            filename = f"{timestamp}_{text_prefix}.wav"
            filepath = output_path / filename

            # 音声ファイルとして保存
            print(f"[{datetime.now().isoformat()}] 音声ファイルを保存中...")
            with open(filepath, "wb") as f:
                f.write(audio_response.content)

            total_time = time.time() - start_time
            print(f"[{datetime.now().isoformat()}] 処理完了")
            print(f"保存先: {filepath}")
            print(f"総処理時間: {total_time:.2f}秒")
            print(f"- TTSリクエスト時間: {request_time:.2f}秒")
            print(f"- 音声データ取得時間: {audio_time:.2f}秒")
            
            return str(filepath), total_time

        except requests.exceptions.RequestException as e:
            print(f"[{datetime.now().isoformat()}] エラーが発生しました: {str(e)}")
            if hasattr(e.response, 'json'):
                print(f"サーバーエラー: {e.response.json()}")
            return None

def main():
    # 使用例
    client = TTSClient()
    text = "こんにちは、テストです。"
    
    result = client.generate_speech(text)
    if result is None:
        print("音声生成に失敗しました")

if __name__ == "__main__":
    main() 