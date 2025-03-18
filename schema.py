from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    """
    TTSリクエストのスキーマ
    
    Attributes:
        text (str): 音声合成するテキスト
        model_id (str): 使用するモデルのID
        speaker_id (str): 話者ID
    """
    text: str = Field(..., description="音声合成するテキスト", max_length=1000)
    model_id: str = Field(
        default="uranai-emotion2",
        description="使用するモデルのID"
    )
    speaker_id: str = Field(
        default="0",
        description="話者ID"
    )

class TTSResponse(BaseModel):
    """
    TTSレスポンスのスキーマ
    
    Attributes:
        id (str): 生成された音声のID
        model_id (str): 使用されたモデルのID
        speaker_id (str): 使用された話者ID
    """
    id: str = Field(..., description="生成された音声のID")
    model_id: str = Field(..., description="使用されたモデルのID")
    speaker_id: str = Field(..., description="使用された話者ID")

class ModelConfig(BaseModel):
    """
    TTSモデルの設定スキーマ
    
    Attributes:
        model_id (str): モデルのID
        name (str): モデルの名前
        description (str): モデルの説明
        model_path (str): モデルファイルのパス
        config_path (str): 設定ファイルのパス
        speakers (dict[str, str]): 利用可能な話者IDと名前のマッピング
        style_vectors_path (str | None): スタイルベクトルファイルのパス
    """
    model_id: str = Field(..., description="モデルのID")
    name: str = Field(..., description="モデルの名前")
    description: str = Field(default="", description="モデルの説明")
    model_path: str = Field(..., description="モデルファイルのパス")
    config_path: str = Field(..., description="設定ファイルのパス")
    speakers: dict[str, str] = Field(
        default={"0": "通常"},
        description="利用可能な話者IDと名前のマッピング"
    )
    style_vectors_path: str | None = Field(
        default=None,
        description="スタイルベクトルファイルのパス"
    )

class ModelListResponse(BaseModel):
    """
    利用可能なモデル一覧のレスポンススキーマ
    
    Attributes:
        models (list[ModelConfig]): 利用可能なモデルのリスト
        default_model_id (str): デフォルトのモデルID
    """
    models: list[ModelConfig] = Field(..., description="利用可能なモデルのリスト")
    default_model_id: str = Field(..., description="デフォルトのモデルID")

class ModelSwitchRequest(BaseModel):
    """
    モデル切り替えリクエストのスキーマ
    
    Attributes:
        model_id (str): 切り替え先のモデルID
    """
    model_id: str = Field(..., description="切り替え先のモデルID")

class ModelSwitchResponse(BaseModel):
    """
    モデル切り替えレスポンスのスキーマ
    
    Attributes:
        model_id (str): 切り替えられたモデルID
        message (str): 切り替え結果のメッセージ
    """
    model_id: str = Field(..., description="切り替えられたモデルID")
    message: str = Field(..., description="切り替え結果のメッセージ")

class ErrorResponse(BaseModel):
    """
    エラーレスポンスのスキーマ
    
    Attributes:
        message (str): エラーメッセージ
        error_code (int | None): エラーコード
    """
    message: str = Field(..., description="エラーメッセージ")
    error_code: int | None = Field(None, description="エラーコード") 