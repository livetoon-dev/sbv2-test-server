import os
import uuid
import json
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
import aiofiles
import shutil
from pathlib import Path
from dotenv import load_dotenv
import asyncio

from schema import (
    TTSRequest, TTSResponse, ErrorResponse,
    ModelConfig, ModelListResponse, ModelSwitchRequest, ModelSwitchResponse
)
from inference import TTSModelManager

# .envファイルの読み込み
load_dotenv()

# 環境変数の取得
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGING_FACE_TOKENが設定されていません")

# Hugging Face APIの初期化
hf_api = HfApi(token=HF_TOKEN)

# モデルのキャッシュディレクトリ
MODELS_CACHE_DIR = Path(os.getenv("MODELS_CACHE_DIR", "models_cache"))
MODELS_CACHE_DIR.mkdir(exist_ok=True)

# モデルマネージャーのインスタンス
model_manager: TTSModelManager | None = None

# 音声キャッシュ用の辞書
audio_cache: dict[str, bytes] = {}

async def download_model_files(model_id: str, repo_id: str) -> dict[str, str]:
    """モデルファイルをダウンロード"""
    try:
        # モデルファイルのダウンロード
        model_file = await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename=f"{model_id}.safetensors",
            repo_type="model",
            token=os.getenv("HUGGING_FACE_TOKEN"),
            local_dir=os.getenv("MODELS_CACHE_DIR", "models_cache"),
            local_dir_use_symlinks=False
        )
        
        # 設定ファイルのダウンロード
        config_file = await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename="config.json",
            repo_type="model",
            token=os.getenv("HUGGING_FACE_TOKEN"),
            local_dir=os.getenv("MODELS_CACHE_DIR", "models_cache"),
            local_dir_use_symlinks=False
        )
        
        # スタイルベクトルファイルのダウンロード
        style_vectors_file = await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename="style_vectors.npy",
            repo_type="model",
            token=os.getenv("HUGGING_FACE_TOKEN"),
            local_dir=os.getenv("MODELS_CACHE_DIR", "models_cache"),
            local_dir_use_symlinks=False
        )
        
        return {
            "model_file": model_file,
            "config_file": config_file,
            "style_vectors_file": style_vectors_file
        }
    except Exception as e:
        print(f"モデルファイルのダウンロードに失敗: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフスパンイベントを管理"""
    global model_manager
    model_manager = TTSModelManager()
    
    try:
        # BERTモデルの初期化
        print("BERTモデルを初期化中...")
        if not model_manager.initialize_bert():
            raise RuntimeError("BERTモデルの初期化に失敗しました")
        
        # モデル設定ファイルの読み込み
        models_config_path = os.getenv("MODELS_CONFIG", "models_config.json")
        if os.path.exists(models_config_path):
            print(f"モデル設定ファイル {models_config_path} を読み込み中...")
            async with aiofiles.open(models_config_path, mode='r') as f:
                content = await f.read()
                models_config = json.loads(content)
            
            # 各モデルをダウンロードして初期化
            for config in models_config["models"]:
                try:
                    print(f"モデル {config['model_id']} をダウンロード中...")
                    # Hugging Faceからモデルファイルをダウンロード
                    paths = await download_model_files(
                        repo_id=config["repo_id"],
                        model_id=config["model_id"]
                    )
                    
                    # モデル設定を更新
                    model_config = ModelConfig(
                        model_id=config["model_id"],
                        name=config["name"],
                        description=config.get("description", ""),
                        model_path=paths["model_file"],
                        config_path=paths["config_file"],
                        speakers=config.get("speakers", {"0": "デフォルト"}),
                        style_vectors_path=paths["style_vectors_file"]
                    )
                    
                    # モデルを追加
                    print(f"モデル {config['model_id']} を初期化中...")
                    model_manager.add_model(
                        model_config,
                        set_as_default=config.get("is_default", False)
                    )
                    print(f"モデル {config['model_id']} を正常に読み込みました")
                    
                    # ウォームアップ用の推論を実行
                    print(f"モデル {config['model_id']} のウォームアップを実行中...")
                    warmup_text = "ウォームアップのための短いテキストです。"
                    try:
                        model = model_manager.get_model(config["model_id"])
                        sampling_rate, _ = model.infer(
                            text=warmup_text,
                            speaker_id=0
                        )
                        print(f"モデル {config['model_id']} のウォームアップが完了しました")
                    except Exception as e:
                        print(f"ウォームアップ中にエラーが発生しました: {str(e)}")
                    
                except Exception as e:
                    print(f"モデル {config['model_id']} の読み込みに失敗しました: {str(e)}")
                    continue
        else:
            # デフォルトモデルのダウンロードと初期化
            print("デフォルトモデルをダウンロード中...")
            model_path, config_path, style_vectors_path = await asyncio.to_thread(
                model_manager.download_default_model
            )
            
            model_config = ModelConfig(
                model_id="jvnv-F1-jp",
                name="JVNV F1 Japanese",
                description="Default Japanese TTS model",
                model_path=model_path,
                config_path=config_path,
                speakers={"0": "デフォルト"},
                style_vectors_path=style_vectors_path
            )
            
            print("デフォルトモデルを初期化中...")
            model_manager.add_model(model_config, set_as_default=True)
            print("デフォルトモデルを正常に読み込みました")
            
            # デフォルトモデルのウォームアップ
            print("デフォルトモデルのウォームアップを実行中...")
            warmup_text = "ウォームアップのための短いテキストです。"
            try:
                model = model_manager.get_model("jvnv-F1-jp")
                sampling_rate, _ = model.infer(
                    text=warmup_text,
                    speaker_id=0
                )
                print("デフォルトモデルのウォームアップが完了しました")
            except Exception as e:
                print(f"ウォームアップ中にエラーが発生しました: {str(e)}")
    
    except Exception as e:
        print(f"初期化中にエラーが発生しました: {str(e)}")
        raise
    
    yield
    
    # クリーンアップ処理
    if model_manager is not None:
        for model_id in list(model_manager.models.keys()):
            model_manager.remove_model(model_id)
        print("全てのモデルを解放しました")

app = FastAPI(lifespan=lifespan)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """利用可能なモデルの一覧を取得"""
    global model_manager
    if model_manager is None:
        raise HTTPException(status_code=500, detail="モデルマネージャーが初期化されていません")
    
    return ModelListResponse(
        models=list(model_manager.configs.values()),
        default_model_id=model_manager.default_model_id
    )

@app.post("/models/switch", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest):
    """使用するモデルを切り替え"""
    global model_manager
    if model_manager is None:
        raise HTTPException(status_code=500, detail="モデルマネージャーが初期化されていません")
    
    try:
        model_manager.get_model(request.model_id)  # モデルの存在確認
        return ModelSwitchResponse(
            model_id=request.model_id,
            message=f"モデルを {request.model_id} に切り替えました"
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"モデル {request.model_id} が見つかりません")

@app.post("/tts", response_model=TTSResponse)
async def create_tts(request: TTSRequest):
    """音声を生成"""
    global model_manager
    if model_manager is None:
        raise HTTPException(status_code=500, detail="モデルマネージャーが初期化されていません")
    
    try:
        # モデルの存在確認
        if request.model_id not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"モデル {request.model_id} が見つかりません")
        
        # モデルの取得
        model = model_manager.get_model(request.model_id)
        config = model_manager.get_config(request.model_id)
        
        # 話者IDの検証
        if request.speaker_id not in config.speakers:
            raise HTTPException(
                status_code=400,
                detail=f"話者ID {request.speaker_id} は {config.model_id} では利用できません"
            )
        
        try:
            # 音声合成
            sampling_rate, audio_array = model.infer(
                text=request.text,
                speaker_id=int(request.speaker_id)
            )
            
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("音声データの生成に失敗しました")
            
            # 音声データをバイト列に変換
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sampling_rate, format='WAV', subtype='PCM_16')
            audio_data = buffer.getvalue()
            
            if len(audio_data) == 0:
                raise ValueError("音声データの変換に失敗しました")
            
            # ユニークIDを生成
            audio_id = str(uuid.uuid4())
            
            # キャッシュに保存
            audio_cache[audio_id] = audio_data
            
            return TTSResponse(
                id=audio_id,
                model_id=config.model_id,
                speaker_id=request.speaker_id
            )
            
        except Exception as e:
            print(f"音声生成中にエラーが発生しました: {str(e)}")
            raise HTTPException(status_code=500, detail=f"音声生成に失敗しました: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/{audio_id}")
async def get_audio(audio_id: str):
    """生成された音声データを取得"""
    if audio_id not in audio_cache:
        raise HTTPException(status_code=404, detail="音声が見つかりません")
    
    audio_data = audio_cache[audio_id]
    
    # キャッシュから削除（1回限りの使用）
    del audio_cache[audio_id]
    
    return StreamingResponse(
        io.BytesIO(audio_data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{audio_id}.wav"'
        }
    )

@app.get("/")
async def root():
    return {"message": "Style-Bert-VITS2 TTS Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 