import os
import sys
import torch
import warnings

# === 1. RTX 5060 Ti (Blackwell) 顯卡相容補丁 ===
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0" 
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
warnings.filterwarnings("ignore")

# === 2. 路徑防呆與路徑注入 ===
current_file_path = os.path.abspath(__file__)
engines_dir = os.path.dirname(current_file_path)         # D:\BangDream-TTS-WebUI\engines
project_root = os.path.dirname(engines_dir)             # D:\BangDream-TTS-WebUI
index_tts_path = os.path.join(engines_dir, "index-tts") # D:\BangDream-TTS-WebUI\engines\index-tts

# 將「專案根目錄」與「IndexTTS-2 核心源碼根目錄」注入 sys.path
for path in [project_root, index_tts_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# === 3. 導入底座引擎與核心模組 ===
# 1) BaseTTSEngine：採用標準專案絕對路徑
from engines.base_engine import BaseTTSEngine

# 2) IndexTTS2：針對 IndexTTS-2 模型，正確導入 infer_v2 模組中的 IndexTTS2 類別
from indextts.infer_v2 import IndexTTS2


class IndexTTSEngine(BaseTTSEngine):
    def __init__(self):
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        if self.model is not None: 
            return
        print(f"📡 正在載入 Bilibili IndexTTS-2 引擎 (設備: {self.device})...")
        
        # 依據 Bilibili 官方的 infer_v2.py 構造函數進行加載
        self.model = IndexTTS2(
            cfg_path=os.path.join(model_path, "config.yaml"),
            model_dir=model_path,
            use_fp16=True if "cuda" in self.device else False,
            device=self.device,
            use_cuda_kernel=False,      # 預設防呆關閉
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False     # 若顯卡與虛擬環境支援，後續可自行改為 True 來加速
        )
        print("✅ IndexTTS-2 模型加載成功！")

    def generate(self, text, voice_config, emotion_idx=0, language="zh"):
        prompt_data = voice_config["prompts"][emotion_idx]
        ref_audio_path = os.path.abspath(os.path.join(
            "models", "voices", voice_config["band"], voice_config["character"], prompt_data["audio_path"]
        ))
        
        print(f"🎙️ IndexTTS-2 開始克隆: {voice_config['character']} | 語言: {language}")

        # 使用臨時檔案路徑，讓官方推理模組直接處理輸出
        temp_wav_path = "temp_engine_output.wav"
        
        with torch.no_grad():
            # 調用 IndexTTS-2 的 infer 方法
            self.model.infer(
                spk_audio_prompt=ref_audio_path,
                text=text,
                output_path=temp_wav_path,
                use_emo_text=True,       # 自動調用您目錄中的 Qwen 模型來分析文本情感，使聲音更生動
                verbose=False
            )
        
        # 讀取剛才生成好的音檔並返回給 WebUI
        import torchaudio
        wav, sr = torchaudio.load(temp_wav_path)
        
        return wav, sr

    def release_memory(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()
            print("♻️ 顯存已清理")