import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 確保根目錄在 path 中，方便導入 engines
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(title="BangDream TTS WebUI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 延遲初始化引擎（在 start 事件中）
tts_engine = None


@app.on_event("startup")
async def startup():
    global tts_engine

    # 可透過環境變數調整路徑，若未設置則使用預設相對路徑
    model_path = os.getenv(
        "TTS_MODEL_PATH",
        os.path.join("models", "checkpoints", "Qwen3-TTS"),
    )
    voice_dir = os.getenv(
        "TTS_VOICE_DIR",
        os.path.join("models", "voices"),
    )

    from engines.qwen_engine import Qwen3TTSEngine

    tts_engine = Qwen3TTSEngine(
        model_path=model_path,
        voice_dir=voice_dir,
    )
    tts_engine.load()


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # e.g. "PoppinParty/Mori"
    speed: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.9


@app.get("/")
async def root():
    return {
        "service": "BangDream TTS WebUI API",
        "status": "running",
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    try:
        audio_bytes = tts_engine.generate(
            text=req.text,
            voice=req.voice,
            speed=req.speed,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from fastapi.responses import Response

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
    )


@app.get("/voices")
async def list_voces():
    """
    列出可用的音色資料夾（簡單掃描 voices/ 下的子目錄）。
    """
    voice_dir = os.getenv(
        "TTS_VOICE_DIR",
        os.path.join("models", "voices"),
    )

    if not os.path.isdir(voice_dir):
        return []

    result = []
    for band in sorted(os.listdir(voice_dir)):
        band_path = os.path.join(voice_dir, band)
        if not os.path.isdir(band_path):
            continue
        members = [
            m
            for m in os.listdir(band_path)
            if os.path.isdir(os.path.join(band_path, m))
        ]
        result.append({
            "band": band,
            "members": sorted(members),
        })

    return result