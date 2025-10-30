# -*- coding: utf-8 -*-
"""
互動式脈波模擬器 

© [2025] [TCMPulse.com/ Helen Hua]
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License (CC BY-NC-SA 4.0).

您可自由：
- 分享 (Share) — 複製及發佈本程式
- 修改 (Adapt) — 重混、轉換及依本程式建立衍生作品

惟需遵照下列條件：
- 姓名標示 (BY) — 您必須給予適當表彰 (credit)。
- 非商業性 (NC) — 您不得將本程式用於商業目的 (營利)。
- 相同方式分享 (SA) — 若您重混、轉換、或依本程式建立衍生作品，
  您必須基於同一授權條款來散布您的貢獻作品。

詳情請見: http://creativecommons.org/licenses/by-nc-sa/4.0/

[重要] 本程式碼為公開發布，僅供學術交流與非營利目的使用。
嚴禁任何未經授權的商業行為。

"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, get_window
import sqlite3
import json

DB_NAME = "pulse_presets.db"

# --- (A) 模擬器核心函數 (與 V5 相同) ---

def create_single_beat_v4(
    fs=500, t_period=0.833,
    # G1: 主峰 (Systolic)
    t1_pct=0.15, s1_pct=0.05,
    # G2: 潮汐波 (Tidal / Split-Peak)
    t2_pct=0.25, a2_ratio=0.0, s2_pct=0.06,
    # G3: 重搏波 (Dicrotic)
    t3_pct=0.45, a3_ratio=0.3, s3_pct=0.08,
    # G4: 第三反彈波 (Tertiary)
    t4_pct=0.65, a4_ratio=0.0, s4_pct=0.10,
    # 舒張期
    decay_factor=0.7
):
    """
    使用「四組高斯疊加模型」生成一個單一的心跳波形
    """
    
    num_samples = int(fs * t_period)
    t = np.linspace(0, t_period, num_samples, endpoint=False)
    
    # G1: 主峰 (Systolic Peak) - 幅度(A1) 固定為 1.0 作為基準
    t1, A1, sigma1 = t1_pct * t_period, 1.0, s1_pct * t_period
    G1 = A1 * np.exp(-(t - t1)**2 / (2 * sigma1**2))
    
    # G2: 潮汐波 (Tidal Wave) - 預設幅度(A2)為 0 (關閉)
    t2, A2, sigma2 = t2_pct * t_period, a2_ratio, s2_pct * t_period
    G2 = A2 * np.exp(-(t - t2)**2 / (2 * sigma2**2))

    # G3: 重搏波 (Dicrotic Wave)
    t3, A3, sigma3 = t3_pct * t_period, a3_ratio, s3_pct * t_period
    G3 = A3 * np.exp(-(t - t3)**2 / (2 * sigma3**2))

    # G4: 第三反彈波 (Tertiary Wave) - 預設幅度(A4)為 0 (關閉)
    t4, A4, sigma4 = t4_pct * t_period, a4_ratio, s4_pct * t_period
    G4 = A4 * np.exp(-(t - t4)**2 / (2 * sigma4**2))
    
    # 合成波形並添加舒張期下降趨勢 (Diastolic Runoff)
    diastolic_decay = np.exp(-t / (t_period * decay_factor))
    wave = (G1 + G2 + G3 + G4) * diastolic_decay
    
    return wave

def generate_pulse_wave_data(
    hr_bpm_mean=72.0, 
    pp_mmhg_mean=40.0, 
    sp_mmhg_mean=100.0, 
    hcv_percent=0.0,
    hrv_percent=2.0,
    duration_sec=8.0, 
    fs=500,
    morph_params={}
):
    """
    生成完整的、帶有變異性的模擬脈搏波數據
    """
    
    total_samples = int(duration_sec * fs)
    time_total = np.linspace(0, duration_sec, total_samples, endpoint=False)
    signal = np.zeros(total_samples) + sp_mmhg_mean 
    
    current_time = 0.0
    
    while current_time < duration_sec:
        hrv_scale = hr_bpm_mean * (hrv_percent / 100.0)
        current_hr = np.random.normal(loc=hr_bpm_mean, scale=max(0.1, hrv_scale))
        current_hr = np.clip(current_hr, 40, 180)
        t_period = 60.0 / current_hr
        
        hcv_scale = pp_mmhg_mean * (hcv_percent / 100.0)
        current_pp = np.random.normal(loc=pp_mmhg_mean, scale=max(0.1, hcv_scale))
        current_pp = np.clip(current_pp, 5, 150)
        
        wave_beat = create_single_beat_v4(
            fs, t_period, **morph_params
        )
            
        wave_zeroed = wave_beat - wave_beat.min()
        current_wave_pp = wave_zeroed.max()
        if current_wave_pp > 0:
            scaling_factor = current_pp / current_wave_pp
            scaled_wave_beat = wave_zeroed * scaling_factor
        else:
            scaled_wave_beat = np.zeros_like(wave_beat)
            
        start_idx = int(round(current_time * fs))
        end_idx = start_idx + len(scaled_wave_beat)
        
        if end_idx > total_samples:
            samples_to_fit = total_samples - start_idx
            if samples_to_fit > 0:
                signal[start_idx:] += scaled_wave_beat[:samples_to_fit] 
            break 
        else:
            signal[start_idx:end_idx] += scaled_wave_beat
            
        current_time += t_period
            
    return time_total, signal

# --- (B) 分析儀核心函數 (與 V5 相同) ---

def run_analysis(pressure_data, fs, max_harmonics=11, n_fft=4096):
    """
    對傳入的時域數據執行完整的 SOP 分析 (分割, FFT, Cn/C0, HCV)
    """
    T = 1.0 / fs
    
    valleys, _ = find_peaks(-pressure_data, distance=int(fs * 0.4), prominence=1)
    
    if len(valleys) < 2:
        return None, None 

    num_beats = len(valleys) - 1
    analysis_results = []
    
    search_width_hz = 0.4 

    for i in range(num_beats):
        start_index = valleys[i]
        end_index = valleys[i+1]
        
        segment_unwindowed = pressure_data[start_index:end_index]
        N_segment = len(segment_unwindowed)
        if N_segment == 0:
            continue
            
        C0 = np.mean(segment_unwindowed)
        
        window = get_window('hann', N_segment)
        segment_windowed = (segment_unwindowed - C0) * window 
        
        fft_vals = np.fft.fft(segment_windowed, n=n_fft)
        fft_freq = np.fft.fftfreq(n_fft, T)
        
        fft_mag = np.abs(fft_vals) / np.sum(window) * 2
        
        mask = (fft_freq >= 0) & (fft_freq <= 40)
        freq_pos = fft_freq[mask]
        mag_pos = fft_mag[mask]
        
        beat_data = {'Beat': i + 1, 'C0': C0}
        
        f0 = 1.0 / (N_segment * T) 
        beat_data['f0 (Hz)'] = f0
        
        for h in range(1, max_harmonics + 1):
            target_freq = f0 * h
            if target_freq > 40:
                break
            
            search_mask = (freq_pos >= target_freq - search_width_hz/2) & (freq_pos <= target_freq + search_width_hz/2)
            
            Cn_amp = 0.0
            if np.any(search_mask):
                Cn_amp = np.max(mag_pos[search_mask])
            
            beat_data[f'C{h}_Amp'] = Cn_amp
            if C0 > 0:
                beat_data[f'C{h}/C0 (%)'] = (Cn_amp / C0) * 100
            else:
                beat_data[f'C{h}/C0 (%)'] = np.nan

        analysis_results.append(beat_data)

    if not analysis_results:
        return None, None
        
    df_per_beat = pd.DataFrame(analysis_results)
    
    hcv_results = []
    for n in range(1, max_harmonics + 1):
        col_name = f'C{n}/C0 (%)'
        if col_name in df_per_beat.columns:
            values = df_per_beat[col_name].dropna().values
            mean_val = np.mean(values)
            std_dev_val = np.std(values, ddof=1)
            
            if mean_val == 0 or pd.isna(mean_val) or pd.isna(std_dev_val):
                hcv_val = np.nan
            else:
                hcv_val = (std_dev_val / mean_val) * 100 
                
            hcv_results.append({
                'Harmonic': f'C{n}',
                'Mean (%)': mean_val,
                'Std Dev (%)': std_dev_val,
                'H.C.V. (%)': hcv_val
            })
            
    df_hcv = pd.DataFrame(hcv_results)
    
    return df_per_beat, df_hcv

# --- (D) V7 升級：資料庫 (SQLite) 函數 ---

def get_db_connection():
    """建立並返回一個資料庫連線"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化資料庫，如果表格不存在則建立"""
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS presets (
            name TEXT PRIMARY KEY,
            params_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def load_presets_from_db():
    """從資料庫讀取所有預設組"""
    try:
        conn = get_db_connection()
        presets_db = conn.execute("SELECT * FROM presets").fetchall()
        conn.close()
        
        # 將資料庫格式 (json string) 轉為 session_state 格式 (dict)
        presets_dict = {}
        for row in presets_db:
            presets_dict[row['name']] = json.loads(row['params_json'])
        return presets_dict
    except Exception as e:
        st.error(f"讀取資料庫失敗: {e}")
        return {}

def save_preset_to_db(preset_name, params_dict):
    """將一個預設組儲存到資料庫"""
    try:
        params_json = json.dumps(params_dict) # 將 dict 轉為 json 字串
        conn = get_db_connection()
        conn.execute(
            "INSERT OR REPLACE INTO presets (name, params_json) VALUES (?, ?)",
            (preset_name, params_json)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"儲存脈型失敗: {e}")
        return False

def delete_preset_from_db(preset_name):
    """從資料庫刪除一個預設組"""
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM presets WHERE name = ?", (preset_name,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"刪除脈型失敗: {e}")
        return False

# --- (E) Streamlit 介面佈局 ---

# *** V9 修正：移除 icon=... 參數 ***
st.set_page_config(page_title="脈波模擬分析工作站", layout="wide")
st.title("脈波模擬分析工作站")

# --- V7 初始化：程式啟動時執行一次 ---
if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True
    
if 'presets' not in st.session_state:
    st.session_state.presets = load_presets_from_db()

# 定義所有需要儲存/讀取的參數的 key
PARAM_KEYS = [
    "hr_bpm_mean", "pp_mmhg_mean", "sp_mmhg_mean", "hcv_percent", "hrv_percent",
    "t1_pct", "s1_pct", 
    "a2_ratio", "t2_pct", "s2_pct",
    "a3_ratio", "t3_pct", "s3_pct",
    "a4_ratio", "t4_pct", "s4_pct",
    "decay_factor"
]

# --- V6/V7 儲存/讀取的回調函數 ---
def save_preset():
    """Callback 函數：儲存當前所有拉桿的參數到 DB"""
    preset_name = st.session_state.new_preset_name
    if not preset_name:
        st.toast("錯誤：請輸入脈型名稱！", icon="🚨")
        return
        
    current_params = {key: st.session_state[key] for key in PARAM_KEYS}
    
    if save_preset_to_db(preset_name, current_params):
        st.session_state.presets = load_presets_from_db() # 重新載入
        st.toast(f"成功儲存脈型： {preset_name}", icon="✅")
        st.session_state.new_preset_name = "" # 清空輸入框

def load_preset():
    """Callback 函數：讀取選定的參數並更新所有拉桿"""
    preset_name = st.session_state.selected_preset
    if preset_name == "---":
        return
        
    if preset_name in st.session_state.presets:
        params_to_load = st.session_state.presets[preset_name]
        for key, value in params_to_load.items():
            if key in st.session_state:
                st.session_state[key] = value
        st.toast(f"成功讀取脈型： {preset_name}", icon="✅")
    else:
        st.toast(f"錯誤：找不到脈型 {preset_name}", icon="🚨")

def delete_preset():
    """Callback 函數：刪除選定的參數"""
    preset_name = st.session_state.selected_preset
    if preset_name == "---":
        st.toast("錯誤：請先選擇一個要刪除的脈型。", icon="🚨")
        return

    if preset_name in st.session_state.presets:
        if delete_preset_from_db(preset_name):
            st.session_state.presets = load_presets_from_db() # 重新載入
            st.toast(f"成功刪除脈型： {preset_name}", icon="🗑️")
    else:
        st.toast(f"錯誤：找不到脈型 {preset_name}", icon="🚨")


# --- 1. 參數調整 (Sidebar) ---
# *** V9 修正：將 Emoji 放回文字中 ***
st.sidebar.header("🎛️ 脈型庫")

# --- 讀取/刪除區塊 ---
preset_names = ["---"] + list(st.session_state.presets.keys())
st.sidebar.selectbox(
    "讀取已存脈型", 
    options=preset_names, 
    key="selected_preset", 
    on_change=load_preset
)
# *** V9 修正 ***
st.sidebar.button("🗑️ 刪除選取的脈型", on_click=delete_preset, use_container_width=True)

# --- 儲存區塊 ---
st.sidebar.text_input("新脈型名稱", key="new_preset_name", placeholder="例如：脈型一 (弦脈)")
# *** V9 修正 ***
st.sidebar.button("💾 儲存目前參數為新脈型", on_click=save_preset, type="primary", use_container_width=True)

st.sidebar.markdown("---")

# *** V9 修正 ***
st.sidebar.header("📈 基礎生理參數 (模擬)")
hr_bpm_mean = st.sidebar.slider(
    "平均心率 (HR)", 50, 150, 72, 1, key="hr_bpm_mean"
)
pp_mmhg_mean = st.sidebar.slider(
    "平均脈壓差 (PP)", 10, 100, 40, 1, key="pp_mmhg_mean"
)
sp_mmhg_mean = st.sidebar.slider(
    "靜態施壓 (C0 基底)", 60, 200, 100, 1, key="sp_mmhg_mean", 
    help="模擬感測器施加的平均靜態壓力。"
)
hcv_percent = st.sidebar.slider(
    "振幅亂度 (H.C.V. %)", 0.0, 20.0, 0.0, 0.1, key="hcv_percent",
    help="模擬逐拍脈壓差的變異係數。0% 代表振幅固定。"
)
hrv_percent = st.sidebar.slider(
    "心率變異 (HRV %)", 0.0, 10.0, 2.0, 0.1, key="hrv_percent",
    help="模擬逐拍心率的變異係數。0% 代表心率固定。"
)
duration_sec = st.sidebar.slider(
    "模擬時長 (s)", 2, 30, 8, 1, key="duration_sec"
)

st.sidebar.markdown("---")

# --- 進階形態學參數 (Sidebar) ---
# *** V9 修正 ***
st.sidebar.header("🔬 型態模擬參數")

# --- G1: 主峰 (Systolic) ---
st.sidebar.subheader("G1: 主峰 (Systolic Peak)")
t1_pct = st.sidebar.slider("G1 位置 (t1 %)", 0.10, 0.30, 0.15, 0.01, key="t1_pct")
s1_pct = st.sidebar.slider("G1 寬度 (s1 %)", 0.02, 0.10, 0.05, 0.01, key="s1_pct")

# --- G2: 潮汐波 (Tidal / Split-Peak) ---
st.sidebar.subheader("G2: 潮汐波 (Tidal / Split-Peak)")
a2_ratio = st.sidebar.slider("G2 幅度 (A2/A1)", 0.0, 1.5, 0.0, 0.05, key="a2_ratio")
t2_pct = st.sidebar.slider("G2 位置 (t2 %)", 0.15, 0.40, 0.25, 0.01, key="t2_pct")
s2_pct = st.sidebar.slider("G2 寬度 (s2 %)", 0.02, 0.15, 0.06, 0.01, key="s2_pct")

# --- G3: 重搏波 (Dicrotic) ---
st.sidebar.subheader("G3: 重搏波 (Dicrotic Wave)")
a3_ratio = st.sidebar.slider("G3 幅度 (A3/A1)", 0.0, 1.0, 0.3, 0.01, key="a3_ratio")
t3_pct = st.sidebar.slider("G3 位置 (t3 %)", 0.30, 0.60, 0.45, 0.01, key="t3_pct")
s3_pct = st.sidebar.slider("G3 寬度 (s3 %)", 0.02, 0.20, 0.08, 0.01, key="s3_pct")

# --- G4: 第三反彈波 (Tertiary) ---
st.sidebar.subheader("G4: 第三反彈波 (Tertiary)")
a4_ratio = st.sidebar.slider("G4 幅度 (A4/A1)", 0.0, 0.5, 0.0, 0.01, key="a4_ratio")
t4_pct = st.sidebar.slider("G4 位置 (t4 %)", 0.40, 0.80, 0.65, 0.01, key="t4_pct")
s4_pct = st.sidebar.slider("G4 寬度 (s4 %)", 0.02, 0.20, 0.10, 0.01, key="s4_pct")

# --- 舒張期 (Diastolic) ---
st.sidebar.subheader("舒張期 (Diastolic)")
decay_factor = st.sidebar.slider("舒張期衰減率 (k)", 0.3, 2.0, 0.7, 0.05, key="decay_factor")

# 儲存形態學參數
morph_params = {
    "t1_pct": t1_pct, "s1_pct": s1_pct,
    "t2_pct": t2_pct, "a2_ratio": a2_ratio, "s2_pct": s2_pct,
    "t3_pct": t3_pct, "a3_ratio": a3_ratio, "s3_pct": s3_pct,
    "t4_pct": t4_pct, "a4_ratio": a4_ratio, "s4_pct": s4_pct,
    "decay_factor": decay_factor
}

# --- (F) 主面板 (Main Panel) ---

# --- 3. 生成數據與繪圖 ---
time_data, pressure_data = generate_pulse_wave_data(
    hr_bpm_mean=hr_bpm_mean,
    pp_mmhg_mean=pp_mmhg_mean,
    sp_mmhg_mean=sp_mmhg_mean, 
    hcv_percent=hcv_percent,
    hrv_percent=hrv_percent,
    duration_sec=duration_sec,
    fs=500,
    morph_params=morph_params
)

# *** V9 修正 ***
st.header("📈 (A) 合成波形圖 (前 4 秒)")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(time_data, pressure_data, linewidth=1.2, color='blue')
ax.set_title(f"Simulated Pulse Wave (HR: {hr_bpm_mean:.0f}, PP: {pp_mmhg_mean:.0f}, C0 Base: {sp_mmhg_mean})")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pressure (mmHg)')
ax.set_xlim(0, min(4, duration_sec))
ax.set_ylim(sp_mmhg_mean - pp_mmhg_mean * 0.2, sp_mmhg_mean + pp_mmhg_mean * 1.3)
ax.grid(True)
st.pyplot(fig)

st.subheader("時域數據下載")
df_simulated = pd.DataFrame({'time': time_data, 'pressure': pressure_data})

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig') 

csv_time_data = convert_df_to_csv(df_simulated)
# *** V9 修正 ***
st.download_button(
    label="📥 下載「時域」數據 (.csv)",
    data=csv_time_data,
    file_name=f"simulated_time_domain_data_hr{hr_bpm_mean}.csv",
    mime='text/csv',
    key='download_time_domain'
)

st.markdown("---")

# --- 4. 分析功能 ---
# *** V9 修正 ***
st.header("🔬 (B) 模擬波形分析")
st.write("按下按鈕，對上方生成的波形執行SOP (逐拍分割、FFT、Cn/C0、HCV)。")

# *** V9 修正 ***
if st.button("▶️ 開始分析此模擬波形 (Run SOP Analysis)"):
    with st.spinner("正在執行 SOP 分析... (逐拍分割、FFT、計算 Cn/C0 與 H.C.V.)..."):
        df_per_beat, df_hcv = run_analysis(pressure_data, fs=500)
        
        if df_per_beat is None or df_hcv is None:
            st.error("分析失敗。可能是模擬時長太短或波形無法辨識。請嘗試增加「模擬時長 (s)」。")
        else:
            st.subheader("SOP 分析報告 (1): 逐拍諧波指數 ($C_n/C_0$)")
            
            beat_cols_to_format = [col for col in df_per_beat.columns if col != 'Beat' and df_per_beat[col].dtype == 'float64']
            format_dict_beat = {col: "{:.3f}" for col in beat_cols_to_format}
            st.dataframe(df_per_beat.style.format(format_dict_beat))
            
            csv_per_beat = convert_df_to_csv(df_per_beat)
            # *** V9 修正 ***
            st.download_button(
                label="📥 下載「逐拍諧波報告」(.csv)",
                data=csv_per_beat,
                file_name="analysis_per_beat_report.csv",
                mime='text/csv',
                key='download_per_beat'
            )

            st.subheader("SOP 分析報告 (2): 諧波亂度 (H.C.V.)")
            
            hcv_cols_to_format = [col for col in df_hcv.columns if col != 'Harmonic' and df_hcv[col].dtype == 'float64']
            format_dict_hcv = {col: "{:.3f}" for col in hcv_cols_to_format}
            st.dataframe(df_hcv.style.format(format_dict_hcv))

            csv_hcv = convert_df_to_csv(df_hcv)
            # *** V9 修正 ***
            st.download_button(
                label="📥 下載「諧波亂度報告」(.csv)",
                data=csv_hcv,
                file_name="analysis_hcv_report.csv",
                mime='text/csv',
                key='download_hcv'
            )