import os
import sys
import argparse


def default_host():
    return "0.0.0.0"


def default_port():
    return 7860


def main():
    parser = argparse.ArgumentParser(
        description="BangDream TTS WebUI"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Listen host (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Listen port (default: 7860)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Qwen3-TTS model directory",
    )

    parser.add_argument(
        "--voice-dir",
        type=str,
        default=None,
        help="Path to voices directory",
    )

    args = parser.parse_args()

    # 設定環境變數，供 api_server.py 使用
    if args.model_path:
        os.environ["TTS_MODEL_PATH"] = args.model_path
    else:
        # 預設相對於 main.py 所在位置
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ.setdefault(
            "TTS_MODEL_PATH",
            os.path.join(base_dir, "models", "checkpoints", "Qwen3-TTS"),
        )

    if args.voice_dir:
        os.environ["TTS_VOICE_DIR"] = args.voice_dir
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ.setdefault(
            "TTS_VOICE_DIR",
            os.path.join(base_dir, "models", "voices"),
        )

    host = args.host or default_host()
    port = args.port or default_port()

    print("=" * 40)
    print("BangDream TTS WebUI")
    print("=" * 40)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"TTS_MODEL_PATH: {os.environ['TTS_MODEL_PATH']}")
    print(f"TTS_VOICE_DIR:  {os.environ['TTS_VOICE_DIR']}")
    print("=" * 40)

    # 使用 uvicorn 啟動 FastAPI
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed.")
        print("You can install it via: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "backend.api_server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()