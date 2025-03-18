import os
import re
import torch
import numpy as np
from pathlib import Path
from huggingface_hub import list_repo_files
from schema import ModelConfig
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

class StyleBertVITS2:
    """Style-Bert-VITS2モデルのラッパークラス"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        style_vectors_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        モデルを初期化します
        
        Args:
            model_path (str): モデルファイルのパス
            config_path (str): 設定ファイルのパス
            style_vectors_path (str): スタイルベクトルファイルのパス
            device (str): 使用するデバイス（"cuda" or "cpu"）
        """
        self.device = device
        self.model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vectors_path,
            device=device
        )
    
    def infer(
        self,
        text: str,
        speaker_id: str = "0",
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0
    ) -> np.ndarray:
        """
        テキストから音声を生成します
        
        Args:
            text (str): 音声合成するテキスト
            speaker_id (str): 話者ID
            noise_scale (float): ノイズスケール（音声の品質に影響）
            noise_scale_w (float): ノイズスケールW（表現力に影響）
            length_scale (float): 長さスケール（話速に影響）
            
        Returns:
            np.ndarray: 生成された音声データ（float32の配列）
        """
        # パラメータのバリデーション
        if not isinstance(speaker_id, str) or not speaker_id.isdigit():
            raise ValueError("話者IDは数字の文字列である必要があります")

        # TTSModelのinferメソッドを呼び出し
        sr, audio = self.model.infer(
            text=text,
            speaker_id=int(speaker_id)
        )
        return audio

class TTSModelManager:
    """TTSモデルを管理するクラス"""
    
    def __init__(self):
        self.models: dict[str, TTSModel] = {}
        self.configs: dict[str, ModelConfig] = {}
        self.default_model_id: str | None = None
        self._bert_initialized: bool = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    @classmethod
    def initialize_bert(cls):
        """BERTモデルを初期化"""
        try:
            bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            return True
        except Exception as e:
            print(f"BERTモデルの初期化に失敗: {str(e)}")
            return False

    def ensure_bert_initialized(self):
        """BERTモデルが初期化されていることを確認"""
        if not self._bert_initialized:
            self._bert_initialized = self.initialize_bert()
        if not self._bert_initialized:
            raise RuntimeError("BERTモデルが初期化されていません")

    @classmethod
    def download_default_model(cls) -> tuple[str, str, str]:
        """デフォルトモデルをダウンロード"""
        from huggingface_hub import hf_hub_download
        import os

        model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
        config_file = "jvnv-F1-jp/config.json"
        style_file = "jvnv-F1-jp/style_vectors.npy"
        
        model_dir = Path("model_assets")
        model_dir.mkdir(exist_ok=True)

        downloaded_files = {}
        for file in [model_file, config_file, style_file]:
            try:
                path = hf_hub_download(
                    "litagin/style_bert_vits2_jvnv",
                    file,
                    local_dir=model_dir
                )
                downloaded_files[file] = path
            except Exception as e:
                print(f"ファイル {file} のダウンロードに失敗: {str(e)}")
                raise

        return (
            downloaded_files[model_file],
            downloaded_files[config_file],
            downloaded_files[style_file]
        )

    def add_model(self, config: ModelConfig, set_as_default: bool = False) -> None:
        """モデルを追加"""
        self.ensure_bert_initialized()
        try:
            model = TTSModel(
                model_path=config.model_path,
                config_path=config.config_path,
                style_vec_path=config.style_vectors_path,
                device=self.device  # 適切なデバイスを使用
            )
            self.models[config.model_id] = model
            self.configs[config.model_id] = config
            
            if set_as_default or self.default_model_id is None:
                self.default_model_id = config.model_id
                
        except Exception as e:
            print(f"モデル {config.model_id} の読み込みに失敗: {str(e)}")
            raise

    def remove_model(self, model_id: str) -> None:
        """モデルを削除"""
        if model_id in self.models:
            del self.models[model_id]
            del self.configs[model_id]
            
            if self.default_model_id == model_id:
                self.default_model_id = next(iter(self.models.keys())) if self.models else None

    def get_model(self, model_id: str | None = None) -> TTSModel:
        """モデルを取得"""
        if model_id is None:
            model_id = self.default_model_id
        if model_id is None or model_id not in self.models:
            raise KeyError(f"モデル {model_id} が見つかりません")
        return self.models[model_id]

    def get_config(self, model_id: str | None = None) -> ModelConfig:
        """モデル設定を取得"""
        if model_id is None:
            model_id = self.default_model_id
        if model_id is None or model_id not in self.configs:
            raise KeyError(f"モデル {model_id} の設定が見つかりません")
        return self.configs[model_id]

    def infer(
        self,
        text: str,
        model_id: str | None = None,
        speaker_id: str = "0"
    ) -> tuple[int, np.ndarray]:
        """音声合成を実行
        
        Args:
            text (str): 音声合成するテキスト
            model_id (str | None): 使用するモデルのID
            speaker_id (str): 話者ID
            
        Returns:
            tuple[int, np.ndarray]: (サンプリングレート, 音声データ)
        """
        model = self.get_model(model_id)
        # TTSModelのinferメソッドはサンプリングレートと音声データのタプルを返す
        sampling_rate, audio_data = model.infer(
            text=text,
            speaker_id=int(speaker_id)
        )
        return sampling_rate, audio_data

    def find_latest_model(self, repo_id: str) -> tuple[str, int, int]:
        """最新のモデルファイルを探す"""
        try:
            # リポジトリ内のファイル一覧を取得
            files = list_repo_files(repo_id, repo_type="model")
            
            # .safetensorsファイルをフィルタリング
            model_files = [f for f in files if f.endswith('.safetensors')]
            
            if not model_files:
                raise ValueError(f"リポジトリ {repo_id} にモデルファイルが見つかりません")
            
            # ファイル名から最新のモデルを特定（エポック数とステップ数が最大のもの）
            latest_file = max(model_files, key=lambda x: (
                int(x.split('_e')[1].split('_')[0]),  # エポック数
                int(x.split('_s')[1].split('.')[0])    # ステップ数
            ))
            
            # ファイル名から情報を抽出
            parts = latest_file.split('_')
            epoch = int(parts[1].replace('e', ''))
            step = int(parts[2].replace('s', '').replace('.safetensors', ''))
            
            return latest_file, epoch, step
            
        except Exception as e:
            raise RuntimeError(f"最新のモデルファイルの検索に失敗しました: {str(e)}") 