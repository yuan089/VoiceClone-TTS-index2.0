import os
import sys
import re
import types
from types import ModuleType

# === 記憶體欺騙補丁 ===
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
# =========================================================================

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

sys.path.append(os.path.abspath("./engines/index-tts"))

import gradio as gr
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import json
import traceback
import chinese_converter
from peft import PeftModel

def _safe_load_patch(path_or_tensor, **kwargs):
    if torch.is_tensor(path_or_tensor): return path_or_tensor, 24000
    y, sr = librosa.load(path_or_tensor, sr=kwargs.get('sample_rate', None))
    return torch.from_numpy(y).unsqueeze(0), sr
torchaudio.load = _safe_load_patch

VOICES_ROOT = os.path.abspath("./models/voices")
MODEL_PATH = os.path.abspath("./engines/index-tts/checkpoints")

engine = None
current_lora_char = None 

# ==========================================
# 🔍 核心功能：自動掃描已訓練的 LoRA 模型
# ==========================================
def get_available_loras():
    loras = []
    if not os.path.exists(VOICES_ROOT):
        return loras
    
    bands = [d for d in os.listdir(VOICES_ROOT) if os.path.isdir(os.path.join(VOICES_ROOT, d))]
    for band in bands:
        band_path = os.path.join(VOICES_ROOT, band)
        chars = [d for d in os.listdir(band_path) if os.path.isdir(os.path.join(band_path, d))]
        for char in chars:
            lora_path = os.path.join(band_path, char, "lora")
            # 檢查是否有 lora 資料夾及 safetensors 權重
            if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
                loras.append(f"{band} / {char}")
    return sorted(loras)

def initialize_engine():
    global engine
    if engine is None:
        print("⏳ 正在預先載入 IndexTTS-2 基礎模型至顯示卡... (約需 15~30 秒)")
        from engines.index_engine import IndexTTSEngine
        engine = IndexTTSEngine()
        engine.load_model(MODEL_PATH)
        print("✅ 基礎模型預載完成！")
    return engine

def tts_generate(menu_mode, upload_audio, lora_choice, text, speed_rate, volume_rate, pan_position,
                 emo_method, emo_audio, use_random,
                 happy, angry, sad, afraid, disgusted, melancholic, surprised, calm):
    global current_lora_char
    
    if not text.strip():
        return None, "❌ 台詞不能為空"
        
    try:
        cur_engine = engine 
        voice_ref_path = None
        lora_status_msg = ""
        
        # ==========================================
        # 🎯 路由 1：Zero-Shot 模式
        # ==========================================
        if menu_mode == "🎙️ Zero-Shot":
            if not upload_audio:
                return None, "❌ Zero-Shot 模式請先上傳或錄製一段參考音訊 (Voice Reference)！"
            voice_ref_path = upload_audio
            
            # 強制卸載 LoRA，保證純淨的 Zero-Shot
            if current_lora_char is not None:
                if hasattr(cur_engine.model.gpt, "unload"):
                    cur_engine.model.gpt = cur_engine.model.gpt.unload()
                current_lora_char = None
            lora_status_msg = "🟡 啟用模式：Zero-Shot (任意聲音模仿)"
            
        # ==========================================
        # 🎯 路由 2：LoRA 模式
        # ==========================================
        elif menu_mode == "✨ LoRA":
            if not lora_choice:
                return None, "❌ 請先從下拉選單選擇一個已訓練的 LoRA 模型！"
                
            band, character = lora_choice.split(" / ")
            char_dir = os.path.join(VOICES_ROOT, band, character)
            lora_path = os.path.join(char_dir, "lora")
            
            # 尋找該角色的預設參考音訊
            config_path = os.path.join(char_dir, "config.json")
            voice_config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    voice_config = json.load(f)
                    
            for k in ["voice_reference", "ref_audio", "ref_audio_path", "prompt_wav", "reference_wav", "ref_wav"]:
                if k in voice_config:
                    potential_path = voice_config[k]
                    if not os.path.isabs(potential_path):
                        path_in_char_dir = os.path.join(char_dir, potential_path)
                        if os.path.exists(path_in_char_dir):
                            voice_ref_path = os.path.abspath(path_in_char_dir)
                            break
                    elif os.path.exists(potential_path):
                        voice_ref_path = potential_path
                        break
            
            # 備用參考音訊搜尋
            if voice_ref_path is None:
                supported_formats = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma")
                audio_files = [f for f in os.listdir(char_dir) if f.lower().endswith(supported_formats)]
                if audio_files:
                    voice_ref_path = os.path.abspath(os.path.join(char_dir, audio_files[0]))
                else:
                    raise FileNotFoundError(f"❌ 錯誤：在目錄 {char_dir} 下找不到該角色的參考音訊！")

            # 掛載 LoRA
            if current_lora_char != lora_choice:
                if hasattr(cur_engine.model.gpt, "unload"):
                    cur_engine.model.gpt = cur_engine.model.gpt.unload()
                cur_engine.model.gpt = PeftModel.from_pretrained(cur_engine.model.gpt, lora_path)
                current_lora_char = lora_choice
                
            lora_status_msg = f"🟢 啟用模式：LoRA 極致還原 [{lora_choice}]"
        
        # ==========================================
        # 🎯 共用語音生成與 ASMR 邏輯
        # ==========================================
        emo_vector, emo_ref_audio, emo_alpha = None, None, 1.0
        if emo_method == "Use emotion vectors":
            emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        elif emo_method == "Use emotion reference audio":
            if emo_audio: emo_ref_audio = emo_audio
            else: return None, "❌ 請上傳情感參考音訊！"
            
        # 原生語速控制補丁
        lr_module = cur_engine.model.s2mel.models['length_regulator']
        if not hasattr(lr_module, "_original_forward"):
            lr_module._original_forward = lr_module.forward

        def patched_lr_forward(self, S_infer, ylens=None, **kwargs):
            if ylens is not None:
                ylens = (ylens.float() / speed_rate).long()
            return self._original_forward(S_infer, ylens=ylens, **kwargs)
            
        lr_module.forward = types.MethodType(patched_lr_forward, lr_module)

        # 處理省略號停頓
        text = text.replace("…", "...").replace("。。", "..")
        segments = re.split(r'(\.{2,})', text)
        
        audio_pieces = []
        final_sr = 22050 
        
        for i, seg in enumerate(segments):
            if i % 2 == 0:
                clean_text = seg.strip()
                if not clean_text: continue
                clean_text = chinese_converter.to_simplified(clean_text)
                
                result = cur_engine.model.infer(
                    spk_audio_prompt=voice_ref_path,
                    text=clean_text,
                    output_path=None, 
                    emo_audio_prompt=emo_ref_audio,
                    emo_alpha=emo_alpha,
                    emo_vector=emo_vector,
                    use_random=use_random,
                    verbose=False
                )
                sr, wav_data = result
                final_sr = sr
                audio_pieces.append(wav_data.flatten())
            else:
                pause_sec = len(seg) * 0.1
                pause_samples = int(pause_sec * final_sr)
                silence = np.zeros(pause_samples, dtype=np.int16)
                audio_pieces.append(silence)
                
        if not audio_pieces:
            return None, "❌ 生成失敗，無效的台詞"
            
        final_wav = np.concatenate(audio_pieces)
            
        # 音量處理
        if volume_rate != 1.0:
            final_wav = final_wav * volume_rate
            if np.abs(final_wav).max() > 1.0:
                final_wav = final_wav / np.abs(final_wav).max()

        # 空間聲學處理
        if pan_position != 0.0:
            angle = (pan_position + 1.0) * (np.pi / 4.0) 
            left_channel = final_wav * np.cos(angle)
            right_channel = final_wav * np.sin(angle)
            stereo_wav = np.column_stack((left_channel, right_channel))
        else:
            stereo_wav = np.column_stack((final_wav, final_wav))

        output_path = "temp_output.wav"
        if os.path.exists(output_path):
            os.remove(output_path)
        sf.write(output_path, stereo_wav, final_sr)
        
        final_log = f"✅ 合成完畢！\n{lora_status_msg}\n驅動硬體: {torch.cuda.get_device_name(0)}"
        return output_path, final_log
        
    except Exception as e:
        return None, f"💥 錯誤: {str(e)}\n{traceback.format_exc()}"


initialize_engine()
available_loras = get_available_loras()

# ==========================================
# UI 佈局設計 (完美重現 Sidebar 側邊選單)
# ==========================================
with gr.Blocks(title="BanG Dream! AI Voice Studio") as demo:
    gr.Markdown("114514")
    
    with gr.Row():
        # ========== 左側導航欄 (Sidebar) ==========
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🗂️ 模式選擇")
            menu_mode = gr.Radio(
                choices=["🎙️ Zero-Shot", "✨ LoRA"], 
                value="🎙️ Zero-Shot", 
                label="",
            )
            
            gr.Markdown("---")
            with gr.Accordion("声音微调(测试版)", open=False):
                speed_rate = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="語速 (Native Speed)")
                volume_rate = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="音量倍率 (Volume)")
                pan_position = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="空間方位 [-1左, 1右]")

        # ========== 右側主內容區 (Main Content) ==========
        with gr.Column(scale=4):
            gr.Markdown("### 語音來源設定 (Speech Synthesis)")
            
            # 動態區塊 1: Zero-Shot 音訊上傳區
            with gr.Group(visible=True) as zs_panel:
                upload_audio = gr.Audio(label="拖放音訊至此處 (Voice Reference)", type="filepath")
                
            # 動態區塊 2: LoRA 模型下拉選單
            with gr.Group(visible=False) as lora_panel:
                if available_loras:
                    lora_choice = gr.Dropdown(choices=available_loras, value=available_loras[0], label="模型列表 (.safetensors)")
                else:
                    lora_choice = gr.Dropdown(choices=[], label="模型列表 (.safetensors) - 尚未找到任何已訓練的 LoRA")

            # 共通組件區
            text_in = gr.Textbox(label="Text (支援省略號停頓 ...)一個省略號0.1秒停頓", lines=4, placeholder="請輸入要合成的台詞...")
            
            with gr.Row():
                btn = gr.Button("🚀 Synthesize (立即合成)", variant="primary", scale=2)
                audio_out = gr.Audio(label="Synthesis Result", type="filepath", scale=3)
            
            status_out = gr.Textbox(label="系統狀態", lines=2)
            
            with gr.Accordion("⚙️ Settings (高級情感控制)", open=False):
                emo_method = gr.Radio(
                    choices=["Same as the voice reference", "Use emotion reference audio", "Use emotion vectors"],
                    value="Same as the voice reference", label="Emotion control method"
                )
                use_random = gr.Checkbox(value=False, label="Randomize emotion sampling")
                emo_audio = gr.Audio(label="上傳情感參考音訊", type="filepath", visible=False)
                
                with gr.Column(visible=False) as vector_group:
                    with gr.Row():
                        happy = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Happy")
                        angry = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Angry")
                    with gr.Row():
                        sad = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Sad")
                        afraid = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Afraid")
                    with gr.Row():
                        disgusted = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Disgusted")
                        melancholic = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Melancholic")
                    with gr.Row():
                        surprised = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Surprised")
                        calm = gr.Slider(0, 1.0, value=0.0, step=0.01, label="Calm")

    # --- 互動邏輯綁定 ---
    # 1. 左側選單切換，動態顯示/隱藏右上角的控制面板
    def toggle_panels(mode):
        if mode == "🎙️ Zero-Shot":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
            
    menu_mode.change(toggle_panels, inputs=menu_mode, outputs=[zs_panel, lora_panel])
    
    # 2. 情感設定面板連動
    def on_emo_method_change(method):
        if method == "Use emotion reference audio": return gr.update(visible=True), gr.update(visible=False)
        elif method == "Use emotion vectors": return gr.update(visible=False), gr.update(visible=True)
        else: return gr.update(visible=False), gr.update(visible=False)
    emo_method.change(on_emo_method_change, inputs=emo_method, outputs=[emo_audio, vector_group])
    
    # 3. 合成按鈕
    btn.click(
        tts_generate, 
        inputs=[menu_mode, upload_audio, lora_choice, text_in, speed_rate, volume_rate, pan_position,
                emo_method, emo_audio, use_random,
                happy, angry, sad, afraid, disgusted, melancholic, surprised, calm], 
        outputs=[audio_out, status_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7865, inbrowser=True)