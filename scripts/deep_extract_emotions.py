import os
import shutil
import json
import re
from collections import defaultdict

# ============ 路徑配置 ============
SOVITS_ROOT = r"D:\GPT-SoVITS v2 pro plus"
TARGET_VOICE_DIR = os.path.join("models", "voices")
# ================================

# 核心角色字典
CHAR_MAP = {
    "仁菜": ("GBC", "Iseri_Nina"), "純田真奈": ("Sumimi", "Sumida_Mana"),
    "高松灯": ("MyGO", "Takamatsu_Tomori"), "长崎素世": ("MyGO", "Nagasaki_Soyo"),
    "椎名立希": ("MyGO", "Shiina_Taki"), "千早爱音": ("MyGO", "Chihaya_Anon"),
    "要乐奈": ("MyGO", "Kaname_Raana"), 
    "祥子": ("AveMujica", "Togawa_Sakiko"), "若叶睦": ("AveMujica", "Wakaba_Mutsumi"),
    "墨提斯": ("AveMujica", "Wakaba_Mutsumi"), "海铃": ("AveMujica", "Yahata_Umiri"), 
    "初华": ("AveMujica", "Misumi_Uika"), "喵梦": ("AveMujica", "Yuutenji_Nyamu"), 
    "弦卷心": ("HelloHappy", "Tsurumaki_Kokoro"), "奥泽美咲": ("HelloHappy", "Okusawa_Misaki"),
    "松原花音": ("HelloHappy", "Matsubara_Kanon"), "濑田薰": ("HelloHappy", "Seta_Kaoru"),
    "北泽育美": ("HelloHappy", "Kitazawa_Hagumi"),
    "丸山彩": ("PastelPalettes", "Maruyama_Aya"), "冰川日菜": ("PastelPalettes", "Hikawa_Hina"),
    "大和麻弥": ("PastelPalettes", "Yamato_Maya"), "白鷺千聖": ("PastelPalettes", "Shirasagi_Chisato"),
    "若宮伊芙": ("PastelPalettes", "Wakamiya_Eve"),
    "湊友希那": ("Roselia", "Minato_Yukina"), "冰川纱夜": ("Roselia", "Hikawa_Sayo"),
    "今井莉莎": ("Roselia", "Imai_Lisa"), "宇田川亚子": ("Roselia", "Udagawa_Ako"),
    "白金燐子": ("Roselia", "Shirokane_Rinko"),
    "美竹蘭": ("Afterglow", "Mitake_Ran"), "青葉モカ": ("Afterglow", "Aoba_Moca"),
    "上原ひまり": ("Afterglow", "Uehara_Himari"), "宇田川巴": ("Afterglow", "Udagawa_Tomoe"),
    "羽沢鶫": ("Afterglow", "Hazawa_Tsugumi"),
    "LAYER": ("RAS", "Wakana_Rei"), "LOCK": ("RAS", "Asahi_Rokka"),
    "MASKING": ("RAS", "Satou_Masuki"), "PAREO": ("RAS", "Nyubara_Reona"),
    "CHU²": ("RAS", "Tamade_Chiyu"),
    "倉田真白": ("Morfonica", "Kurata_Mashiro"), "桐ヶ谷透子": ("Morfonica", "Kirigaya_Touko"),
    "廣町七深": ("Morfonica", "Hiromachi_Nanami"), "二葉筑紫": ("Morfonica", "Futaba_Tsukushi"),
    "八潮瑠唯": ("Morfonica", "Yashio_Rui"),
    "戸山香澄": ("PoppinParty", "Toyama_Kasumi"), "市谷有咲": ("PoppinParty", "Ichigaya_Arisa"),
    "市ヶ谷有咲": ("PoppinParty", "Ichigaya_Arisa"), "花园多惠": ("PoppinParty", "Hanazono_Tae"),
    "牛込里美": ("PoppinParty", "Ushigome_Rimi"), "山吹沙绫": ("PoppinParty", "Yamabuki_Saaya"),
}

def identify_character(path_str):
    for zh_name, (band, eng_name) in CHAR_MAP.items():
        if zh_name in path_str:
            return band, eng_name
    return None, None

def extract_emotion_and_text(filename, dirpath):
    """從檔名和路徑中提取情緒標籤與文本"""
    name_without_ext = os.path.splitext(filename)[0]
    emotion = "default"
    text = name_without_ext

    # 1. 解析檔名開頭的括號 (例如: (低落)でもそういう... -> emotion: 低落)
    match = re.match(r'^[\(（【](.*?)[\)）】](.*)', name_without_ext)
    if match:
        emotion = match.group(1).strip()
        text = match.group(2).strip()
    
    # 2. 如果檔名沒有標籤，我們看看資料夾名稱有沒有暗示 (例如 "大魔姬", "soyo0", "可爱捏")
    else:
        folder_name = os.path.basename(dirpath)
        if any(keyword in folder_name for keyword in ["夹", "大魔姬", "可爱捏", "软糯", "吐槽", "白", "黑", "乖猫", "哈气"]):
            emotion = folder_name.replace("丰川祥子", "").replace("喵梦", "").replace("市谷有咲", "").replace("长崎素世", "").strip("（）()_ ")
            if not emotion: emotion = "special"

    return emotion, text

if __name__ == "__main__":
    print("⛏️ 開始掘地三尺！全面收集角色多情緒語音數據...")
    
    # 用於儲存每個角色的所有語音: char_db["Takamatsu_Tomori"] = [{"emotion": "...", "audio": "...", "text": "..."}]
    char_db = defaultdict(list)
    char_band_map = {}
    
    # 統計用
    total_files = 0

    for dirpath, _, filenames in os.walk(SOVITS_ROOT):
        audio_files = [f for f in filenames if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for audio in audio_files:
            emotion, text = extract_emotion_and_text(audio, dirpath)
            
            # 過濾掉非台詞的短檔名 (如 "aya.mp3")，設定至少 5 個字元
            if len(text) < 5: continue
            
            band, std_name = identify_character(dirpath)
            if not band: continue
            
            # 防重複機制 (同一個角色，同一句話不重複收錄)
            if any(p["reference_text"] == text for p in char_db[std_name]):
                continue

            char_band_map[std_name] = band
            
            # 建立目標目錄
            target_dir = os.path.join(TARGET_VOICE_DIR, band, std_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # 為了避免檔名衝突，我們用流水號命名音檔 (例: Takamatsu_Tomori_01.wav)
            file_idx = len(char_db[std_name]) + 1
            ext = os.path.splitext(audio)[1]
            new_audio_name = f"{std_name}_{file_idx:02d}{ext}"
            
            # 複製檔案
            shutil.copy2(os.path.join(dirpath, audio), os.path.join(target_dir, new_audio_name))
            
            # 記錄到資料庫
            char_db[std_name].append({
                "emotion": emotion,
                "audio_path": new_audio_name,
                "reference_text": text
            })
            
            total_files += 1
            print(f"✅ 發現 [{std_name}] 語音 | 情緒: {emotion} | 💬 {text[:15]}...")

    # 掃描完畢，為每個角色生成一個綜合的 config.json
    print("\n📝 正在為角色生成綜合設定檔 (config.json)...")
    for std_name, prompts in char_db.items():
        band = char_band_map[std_name]
        target_dir = os.path.join(TARGET_VOICE_DIR, band, std_name)
        
        config = {
            "character": std_name,
            "band": band,
            "prompts": prompts  # 這是一個 List，包含了該角色的所有情緒與對應音檔
        }
        with open(os.path.join(target_dir, "config.json"), 'w', encoding='utf-8') as cf:
            json.dump(config, cf, ensure_ascii=False, indent=2)

    print(f"\n🎉 掘地三尺任務大成功！")
    print(f"📊 總共提取了 {total_files} 段有效台詞，涵蓋了 {len(char_db)} 位角色。")
    print(f"📁 檔案已妥善存放在: {os.path.abspath(TARGET_VOICE_DIR)}")