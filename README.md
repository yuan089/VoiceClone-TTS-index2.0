###venv.zip在realse區域，必須下載完之後解壓放在文件根目錄才可以運行本軟件
# 高度客製化的TTS 語音工作站 (IndexTTS-2 Custom Build)

基於目前業界最先進的零樣本 (Zero-Shot) 語音生成大模型 **IndexTTS-2 (MaskGCT)** 深度客製化的本機部署方案。
也可以專為動漫角色語音克隆打造。我們解決了原版在 Windows 上繁瑣的編譯報錯，並專門開發了**「全自動 LoRA 一鍵訓練」**。

---

##  核心特色 

### 情感控制 WebUI
* 擁有情感起伏。

### 全自動 LoRA 一鍵訓練 (`train_app.py`)
原版 IndexTTS-2 僅支援推理，我們在不破壞原模型的情況下，強行植入了微調管線：
* **免手動標註 (Auto-ASR)**：拖入音訊檔，系統自動呼叫阿里 `SenseVoice` 進行高精度中/日/英聽寫。
* **Windows 完美相容**：獨家採用「記憶體欺騙法」，100% 繞過 `WeTextProcessing` (pynini) 在 Windows 上無法編譯的死亡報錯。
* **MSE Loss 潛在空間對齊**：摒棄傳統的 Token 訓練，直接在模型的 Latent Space 進行 MSE 損失計算，**完美鎖死角色聲線，且絕不喪失語言閱讀能力**。
* **防呆機制**：隨便拖入整個資料夾，系統會自動剔除 `.json`、`.txt` 或圖片等非音訊檔，防崩潰。

---
