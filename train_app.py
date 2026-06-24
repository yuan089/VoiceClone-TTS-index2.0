import os
import sys
import shutil
from types import ModuleType

# === 記憶體欺騙補丁：繞過 pynini 編譯 ===
mock_processor = ModuleType("processor")
class DummyProcessor:
    def __init__(self, *args, **kwargs): pass
    def normalize(self, text, *args, **kwargs): return text
mock_processor.Processor = DummyProcessor
mock_tn = ModuleType("tn")
mock_tn.processor = mock_processor
mock_frontend = ModuleType("frontend")
mock_frontend.tn = mock_tn
mock_wetext = ModuleType("wetextprocessing")
mock_wetext.frontend = mock_frontend
sys.modules["wetextprocessing"] = mock_wetext
sys.modules["wetextprocessing.frontend"] = mock_frontend
sys.modules["wetextprocessing.frontend.tn"] = mock_tn
sys.modules["wetextprocessing.frontend.tn.processor"] = mock_processor
# ========================================

sys.path.append(os.path.abspath("./engines/index-tts"))

import gradio as gr
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import chinese_converter
from funasr import AutoModel
from peft import LoraConfig, get_peft_model
from indextts.infer_v2 import IndexTTS2

# 路徑設定
VOICES_ROOT = os.path.abspath("./models/voices")
MODEL_PATH = os.path.abspath("./engines/index-tts/checkpoints")

# 全域變數
asr_model = None
tts_model = None

def init_models():
    global asr_model, tts_model
    if asr_model is None:
        print("📥 正在載入 SenseVoice 自動聽寫模型...")
        asr_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, device="cuda:0")
    if tts_model is None:
        print("📥 正在載入 IndexTTS-2 基礎模型...")
        tts_model = IndexTTS2(cfg_path=f"{MODEL_PATH}/config.yaml", model_dir=MODEL_PATH)
    return asr_model, tts_model

def train_lora_app(band_name, char_name, audio_files, num_epochs=20):
    if not band_name or not char_name:
        yield "❌ 請輸入樂團與角色名稱！"
        return
    if not audio_files:
        yield "❌ 請拖入至少一個音訊檔案！"
        return

    yield "⏳ 正在初始化模型與建立資料夾..."
    asr, tts = init_models()
    
    char_dir = os.path.join(VOICES_ROOT, band_name, char_name)
    dataset_dir = os.path.join(char_dir, "dataset")
    lora_dir = os.path.join(char_dir, "lora")
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    metadata = {}
    
    yield "🎙️ 正在進行自動語音識別 (ASR) 與文本預處理..."
    # 建立有效的檔案計數器
    valid_count = 0
    for file_obj in audio_files:
        # 處理 Gradio 傳入的多檔案物件類型
        if isinstance(file_obj, str):
            tmp_path = file_obj
        elif hasattr(file_obj, "name"):
            tmp_path = file_obj.name
        elif isinstance(file_obj, dict) and "name" in file_obj:
            tmp_path = file_obj["name"]
        else:
            tmp_path = str(file_obj)
            
        ext = os.path.splitext(tmp_path)[-1].lower()
        
        if ext not in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"]:
            print(f"⚠️ 忽略非音訊檔案: {tmp_path}")
            continue
            
        save_name = f"{char_name}_{valid_count:03d}{ext}"
        save_path = os.path.join(dataset_dir, save_name)
        
        shutil.copy(tmp_path, save_path)
        
        # 呼叫 SenseVoice 進行自動聽寫 (支援中/日/英)
        res = asr.generate(input=save_path, language="auto", use_itn=True)[0]
        text = res['text']
        
        # 核心防幻覺：強制繁轉簡，確保詞表能完全識別
        text = chinese_converter.to_simplified(text)
        metadata[save_name] = text
        
        # 順便把第一句話存成參考音訊
        if valid_count == 0:
            ref_save = os.path.join(char_dir, f"ref_{char_name}.wav")
            shutil.copy(save_path, ref_save)
            with open(os.path.join(char_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"ref_audio": f"ref_{char_name}.wav"}, f, ensure_ascii=False, indent=2)
                
        valid_count += 1

    with open(os.path.join(dataset_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    yield f"✅ 資料集處理完成！共 {valid_count} 筆有效資料，即將開始 LoRA 訓練..."

    # --- 開始 LoRA 訓練邏輯 ---
    # 為了防遺忘，凍結基礎參數
    for param in tts.gpt.parameters():
        param.requires_grad = False

    # 防過擬合設定：Rank=4 保持變動微小，不破壞語言能力
    peft_config = LoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn", "c_proj"], lora_dropout=0.1, bias="none"
    )
    
    # 為了能重複訓練，先卸載舊的 Lora (如果有)
    if hasattr(tts.gpt, "peft_config"):
        tts.gpt = tts.gpt.unload()
        
    tts.gpt = get_peft_model(tts.gpt, peft_config)
    tts.gpt.train()
    tts.gpt.to(tts.device)

    # 資料集定義
    class MiniDataset(Dataset):
        def __init__(self, metadata, dataset_dir):
            self.keys = list(metadata.keys())
            self.metadata = metadata
            self.dataset_dir = dataset_dir
        def __len__(self): return len(self.keys)
        def __getitem__(self, idx):
            filename = self.keys[idx]
            text = self.metadata[filename]
            wav_path = os.path.join(self.dataset_dir, filename)
            token_ids = tts.tokenizer.convert_tokens_to_ids(tts.tokenizer.tokenize(text))
            return {"wav_path": wav_path, "token_ids": torch.tensor(token_ids, dtype=torch.long)}

    dataset = MiniDataset(metadata, dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, tts.gpt.parameters()), lr=1e-4)

    torch.set_grad_enabled(True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            token_ids = batch["token_ids"].to(tts.device)
            wav_path = batch["wav_path"][0]
            
            # 提取特徵
            audio, sr = tts._load_and_cut_audio(wav_path, 15)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
            inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                spk_cond_emb = tts.get_emb(inputs["input_features"].to(tts.device), inputs["attention_mask"].to(tts.device))
                quant_result = tts.semantic_codec.quantize(spk_cond_emb)
                codes = quant_result[0] if (quant_result[0].dtype not in [torch.float32, torch.float16] and quant_result[0].ndim <= 2) else quant_result[1]
                codes = codes.long()
            
            # 🎯 [最新版最佳邏輯]：調用官方原生的前向傳播，並使用 MSE Loss 完美對齊潛在空間！
            cond_mel_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=tts.device)
            emo_cond_emb = spk_cond_emb
            emovec = tts.gpt.merge_emovec(
                spk_cond_emb, emo_cond_emb, cond_mel_lengths, cond_mel_lengths, alpha=1.0
            )
            speech_conditioning_latent = tts.gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_mel_lengths)
            use_speed = torch.zeros(spk_cond_emb.size(0), dtype=torch.long, device=tts.device)

            latent = tts.gpt(
                speech_conditioning_latent,
                token_ids,
                torch.tensor([token_ids.shape[-1]], device=tts.device),
                codes,
                torch.tensor([codes.shape[-1]], device=tts.device),
                emo_cond_emb,
                cond_mel_lengths=cond_mel_lengths,
                emo_cond_mel_lengths=cond_mel_lengths,
                emo_vec=emovec,
                use_speed=use_speed,
            )
            
            # 計算 MSE Loss (強迫 LoRA 記憶角色聲線)
            target_emb = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2)
            min_len = min(latent.size(2), target_emb.size(2))
            loss = F.mse_loss(latent[:, :, :min_len], target_emb[:, :, :min_len])
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        yield f"🔥 訓練中... Epoch [{epoch+1}/{num_epochs}] | MSE Loss: {avg_loss:.4f}"

    # 儲存 LoRA
    tts.gpt.save_pretrained(lora_dir)
    tts.gpt = tts.gpt.unload() # 卸載以還原乾淨模型
    yield f"🎉 訓練大功告成！角色 {char_name} 的專屬防幻覺 LoRA 已成功寫入聲庫，現在您可以關閉此視窗，回到 WebUI 使用她了！"

# === Gradio UI 佈局 ===
with gr.Blocks(title="BanG Dream! LoRA Trainer") as app:
    gr.Markdown("")
    
    with gr.Row():
        with gr.Column():
            band_input = gr.Textbox(label="樂團名稱 (例: PastelPalettes)", placeholder="輸入樂團名稱（英文或拼音佳）")
            char_input = gr.Textbox(label="角色名稱 (例: Aya_Maruyama)", placeholder="輸入角色名稱（英文或拼音佳）")
            epochs_slider = gr.Slider(5, 50, value=20, step=1, label="訓練輪數 (Epochs) - 建議20~30防止遺忘")
            
        with gr.Column():
            audio_files = gr.File(label="拖入角色的音訊片段 (支援多選 .wav / .mp3 / 且會自動過濾非音檔)", file_count="multiple")
            train_btn = gr.Button("🚀 一鍵開始煉丹！", variant="primary")
            
    log_output = gr.Textbox(label="訓練日誌", lines=5, interactive=False)
    
    train_btn.click(
        train_lora_app,
        inputs=[band_input, char_input, audio_files, epochs_slider],
        outputs=[log_output]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7866, inbrowser=True)