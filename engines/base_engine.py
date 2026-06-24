# D:\BangDream-TTS-WebUI\engines\base_engine.py

class BaseTTSEngine:
    """
    BanG Dream! TTS 引擎的基礎抽象類別。
    所有 TTS 引擎（如 IndexTTSEngine、QwenTTSEngine 等）都必須繼承此類別並實作其方法。
    """
    def __init__(self):
        self.model = None
        self.device = "cpu"

    def load_model(self, model_path):
        """
        載入模型。需由子類別實作。
        """
        raise NotImplementedError("子類別必須實作 load_model 方法")

    def generate(self, text, voice_config, emotion_idx=0, language="zh"):
        """
        生成語音。需由子類別實作。
        回傳值應為 (wav_tensor_or_numpy, sample_rate)
        """
        raise NotImplementedError("子類別必須實作 generate 方法")

    def release_memory(self):
        """
        釋放顯存與記憶體。需由子類別實作。
        """
        raise NotImplementedError("子類別必須實作 release_memory 方法")