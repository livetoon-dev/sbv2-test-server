import numpy as np
import pytest
from fastapi.testclient import TestClient
import wave
from io import BytesIO
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
from schema import ModelConfig
from huggingface_hub import hf_hub_download
import tempfile
from datetime import datetime

from app import app
from inference import TTSModelManager
from scripts.tts_client import TTSClient

@pytest.fixture
def temp_output_dir():
    """一時的な出力ディレクトリを作成"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(autouse=True)
def setup_test_files():
    """テスト用のファイルを作成"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    repo_id = "LiveTaro/uranai-emotion2"
    
    try:
        # 実際のconfig.jsonをダウンロード
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model",
            local_dir=test_dir,
            local_dir_use_symlinks=False
        )
        
        # スタイルベクトルファイルをダウンロード
        style_vectors_file = hf_hub_download(
            repo_id=repo_id,
            filename="style_vectors.npy",
            repo_type="model",
            local_dir=test_dir,
            local_dir_use_symlinks=False
        )
        
        # ダミーのモデルファイルを作成
        model_path = test_dir / "uranai-emotion2.safetensors"
        model_path.write_bytes(b"dummy_model_data")
        
    except Exception as e:
        pytest.fail(f"テストファイルの設定に失敗しました: {str(e)}")
    
    yield
    
    # クリーンアップ
    if test_dir.exists():
        for file in test_dir.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        test_dir.rmdir()

@pytest.fixture
def mock_tts_model():
    """TTSモデルのモックを作成"""
    mock = MagicMock()
    # 1秒の無音データを生成（float32型）
    audio_data = np.zeros(22050, dtype=np.float32)
    mock.infer.return_value = (22050, audio_data)  # サンプリングレートと音声データのタプルを返す
    return mock

@pytest.fixture
def test_model_manager(mock_tts_model):
    """テスト用のモデルマネージャーを作成"""
    model_manager = TTSModelManager()
    
    # テスト用のモデル設定を追加
    config = ModelConfig(
        model_id="uranai-emotion2",
        name="占い師感情2",
        description="感情表現が豊かな占い師の音声モデル",
        model_path="test_data/uranai-emotion2.safetensors",
        config_path="test_data/config.json",
        speakers={"0": "通常"},
        style_vectors_path="test_data/style_vectors.npy"
    )
    
    # モックモデルを設定
    model_manager.models["uranai-emotion2"] = mock_tts_model
    model_manager.configs["uranai-emotion2"] = config
    model_manager.default_model_id = "uranai-emotion2"
    
    return model_manager

@pytest.fixture
def test_app(test_model_manager):
    """テスト用のアプリケーションを作成"""
    with patch('app.model_manager', test_model_manager):
        return app

@pytest.fixture
def client(test_model_manager):
    """テストクライアントを作成"""
    app.dependency_overrides = {}
    with patch('app.model_manager', test_model_manager):
        with TestClient(app) as client:
            yield client

def test_tts_client_basic(client, temp_output_dir):
    """TTSクライアントの基本機能テスト"""
    tts_client = TTSClient(base_url="http://testserver")
    text = "こんにちは、テストです。"
    
    result = tts_client.generate_speech(
        text=text,
        model_id="uranai-emotion2",
        speaker_id="0",
        output_dir=temp_output_dir
    )
    
    assert result is not None
    filepath, total_time = result
    
    # 出力ファイルの検証
    output_path = Path(filepath)
    assert output_path.exists()
    assert output_path.suffix == ".wav"
    
    # ファイル名のフォーマット検証
    filename = output_path.name
    assert datetime.now().strftime("%Y%m%d") in filename
    assert text[:20].replace(" ", "_").replace("/", "_") in filename
    
    # 処理時間の検証
    assert total_time > 0
    
    # 音声データの検証
    with wave.open(str(output_path), 'rb') as wf:
        assert wf.getnchannels() == 1  # モノラル
        assert wf.getsampwidth() == 2  # 16bit
        assert wf.getframerate() == 22050  # サンプリングレート
        
        frames = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
        assert len(audio_array) > 0

def test_tts_client_invalid_model(client, temp_output_dir):
    """存在しないモデルIDでのクライアントテスト"""
    tts_client = TTSClient(base_url="http://testserver")
    result = tts_client.generate_speech(
        text="こんにちは",
        model_id="non_existent_model",
        speaker_id="0",
        output_dir=temp_output_dir
    )
    assert result is None

def test_tts_client_invalid_speaker(client, temp_output_dir):
    """存在しない話者IDでのクライアントテスト"""
    tts_client = TTSClient(base_url="http://testserver")
    result = tts_client.generate_speech(
        text="こんにちは",
        model_id="uranai-emotion2",
        speaker_id="999",
        output_dir=temp_output_dir
    )
    assert result is None

def test_tts_client_output_directory(client):
    """出力ディレクトリの作成テスト"""
    tts_client = TTSClient(base_url="http://testserver")
    test_output_dir = Path("test_outputs")
    
    try:
        result = tts_client.generate_speech(
            text="こんにちは",
            output_dir=str(test_output_dir)
        )
        
        assert result is not None
        assert test_output_dir.exists()
        assert test_output_dir.is_dir()
        
        filepath, _ = result
        output_file = Path(filepath)
        assert output_file.exists()
        assert output_file.parent == test_output_dir
        
    finally:
        # クリーンアップ
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)

def test_tts_stream(client):
    """TTSのストリーミングテスト実行"""
    
    # TTSリクエストの作成
    request_data = {
        "text": "こんにちは、占い師トーンです。",
        "model_id": "uranai-emotion2",
        "speaker_id": "0",
        "noise_scale": 0.667,
        "noise_scale_w": 0.8,
        "length_scale": 1.0
    }
    
    try:
        # 音声生成リクエスト
        response = client.post("/tts", json=request_data)
        assert response.status_code == 200
        result = response.json()
        
        # レスポンスの検証
        assert "id" in result
        assert "model_id" in result
        assert "speaker_id" in result
        assert result["model_id"] == request_data["model_id"]
        assert result["speaker_id"] == request_data["speaker_id"]
        
        # 音声データのストリーミング取得
        audio_response = client.get(f"/tts/{result['id']}")
        assert audio_response.status_code == 200
        assert audio_response.headers["Content-Type"] == "audio/wav"
        
        # 音声データの検証
        audio_data = BytesIO(audio_response.content)
        with wave.open(audio_data, 'rb') as wf:
            # WAVファイルのフォーマット検証
            assert wf.getnchannels() == 1  # モノラル
            assert wf.getsampwidth() == 2  # 16bit
            assert wf.getframerate() == 22050  # サンプリングレート
            
            # 音声データの長さを検証
            frames = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16)
            assert len(audio_array) > 0
        
        # 2回目のリクエストで404エラーになることを確認（1回限りの使用）
        second_response = client.get(f"/tts/{result['id']}")
        assert second_response.status_code == 404
        
    except Exception as e:
        pytest.fail(f"テストに失敗しました: {str(e)}")

def test_tts_invalid_model(client):
    """存在しないモデルIDでのテスト"""
    request_data = {
        "text": "こんにちは",
        "model_id": "non_existent_model",
        "speaker_id": "0"
    }
    
    response = client.post("/tts", json=request_data)
    assert response.status_code == 404

def test_tts_invalid_speaker(client):
    """存在しない話者IDでのテスト"""
    request_data = {
        "text": "こんにちは",
        "model_id": "uranai-emotion2",
        "speaker_id": "999"  # 存在しない話者ID
    }
    
    response = client.post("/tts", json=request_data)
    assert response.status_code == 400

def test_tts_invalid_audio_id(client):
    """存在しない音声IDでのテスト"""
    response = client.get("/tts/non_existent_audio_id")
    assert response.status_code == 404

def test_model_manager_device_selection():
    """モデルマネージャーのデバイス選択テスト"""
    with patch('torch.cuda.is_available', return_value=True):
        manager = TTSModelManager()
        assert manager.device == "cuda"
    
    with patch('torch.cuda.is_available', return_value=False):
        manager = TTSModelManager()
        assert manager.device == "cpu"

def test_model_initialization_with_device(mock_tts_model):
    """モデル初期化時のデバイス設定テスト"""
    with patch('torch.cuda.is_available', return_value=True):
        manager = TTSModelManager()
        config = ModelConfig(
            model_id="test-model",
            name="Test Model",
            description="Test model for device testing",
            model_path="test_data/model.safetensors",
            config_path="test_data/config.json",
            speakers={"0": "通常"},
            style_vectors_path="test_data/style_vectors.npy"
        )
        manager.add_model(config)
        
        # TTSModelの初期化時に正しいデバイスが渡されたことを確認
        mock_tts_model.assert_called_with(
            model_path=config.model_path,
            config_path=config.config_path,
            style_vec_path=config.style_vectors_path,
            device="cuda"
        ) 