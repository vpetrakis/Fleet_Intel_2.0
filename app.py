import streamlit as st
import pandas as pd
import numpy as np
import re, io, math, traceback, base64, warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# TITAN CORE: v25.0 THE MASTERPIECE (AD-to-AD Tri-State Physics Engine)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from xgboost import XGBRegressor
    import shap
    HAS_ML = True
except ImportError:
    HAS_ML = False

warnings.filterwarnings("ignore")

# UI CONFIGURATION & ASSETS
st.set_page_config(page_title="POSEIDON TITAN", page_icon="⚓", layout="wide", initial_sidebar_state="collapsed")

_LOGO = base64.b64encode(b'<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#c9a84c"/><stop offset="50%" stop-color="#00e0b0"/><stop offset="100%" stop-color="#005f73"/></linearGradient></defs><circle cx="24" cy="24" r="22" fill="none" stroke="url(#pg)" stroke-width="0.8" opacity=".3"/><path d="M24 6L24 42" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"/><path d="M12 24Q24 32 36 24" fill="none" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"/></svg>').decode()

# ANTI-OVERLAP CSS
_CSS = '''<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500;600&family=Hanken+Grotesk:wght@300;400;500;600;700&display=swap');
:root{--bg:#020609;--s1:#080d14;--s2:#0c1219;--b1:rgba(201,168,76,0.06);--b2:rgba(201,168,76,0.15);--b3:rgba(0,224,176,0.12);--acc:#00e0b0;--acc2:#c9a84c;--red:#e63946;--amber:#d4a843;--purple:#7b68ee;--t1:#dce8f0;--t2:#6d8599;--t3:#3a4d5e;--r:12px;--fd:'Bricolage Grotesque',sans-serif;--fb:'Hanken Grotesk',sans-serif;--fm:'Geist Mono',monospace}
html,body,[class*="css"]{font-family:var(--fb)!important;background:var(--bg)!important;color:var(--t1)}
.stApp{background:var(--bg);background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E")}
header,footer,#MainMenu{visibility:hidden!important;display:none!important}
.block-container{padding:0.8rem 2.5rem 0!important;max-width:100%!important}
h1,h2,h3,h4{font-family:var(--fd)!important;font-weight:800!important;color:#fff!important;letter-spacing:-.03em!important}
.hero{background:linear-gradient(135deg,var(--s1),rgba(0,95,115,0.08));border:1px solid var(--b1);border-radius:16px;padding:30px 40px;margin-bottom:24px;display:flex;align-items:center;justify-content:space-between;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent 5%,var(--acc2) 30%,var(--acc) 50%,var(--acc2) 70%,transparent 95%);opacity:.4}
.hero::after{content:'';position:absolute;bottom:0;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,rgba(0,224,176,0.15),transparent)}
.hero-left{display:flex;align-items:center;gap:22px}.hero-logo{width:48px;height:48px;filter:drop-shadow(0 0 12px rgba(0,224,176,0.2))}
.hero-title{font-family:var(--fd);font-weight:800;font-size:1.75rem;letter-spacing:-.04em;background:linear-gradient(135deg,#fff 0%,var(--acc2) 40%,var(--acc) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1}
.hero-sub{font-family:var(--fm);font-size:.58rem;color:var(--t3);text-transform:uppercase;letter-spacing:.2em;font-weight:500;margin-top:4px}
.hero-badge{font-family:var(--fm);font-size:.55rem;color:var(--t3);text-align:right;line-height:2;letter-spacing:.06em}.hero-badge span{color:var(--acc);font-weight:600}
[data-testid="stFileUploader"]{background:var(--s1)!important;border:1px dashed var(--b2)!important;border-radius:var(--r)!important;padding:14px!important;transition:all .4s}
[data-testid="stFileUploader"]:hover{border-color:var(--acc2)!important;box-shadow:0 0 40px rgba(201,168,76,0.06)}
[data-testid="stFileUploader"] *{color:var(--t1)!important;font-family:var(--fb)!important}
[data-testid="stFileUploader"] button{background:rgba(201,168,76,.08)!important;color:var(--acc2)!important;border:1px solid var(--b2)!important;border-radius:8px!important;font-weight:600!important}
div[data-testid="stMetric"]{background:linear-gradient(180deg,var(--s1),var(--s2))!important;border:1px solid var(--b1)!important;border-radius:var(--r);padding:15px 12px!important;position:relative;overflow:hidden;transition:border-color .3s}
div[data-testid="stMetric"]:hover{border-color:var(--b2)!important}
div[data-testid="stMetric"]::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--acc2),transparent);opacity:0;transition:opacity .3s}
div[data-testid="stMetricLabel"]{font-size:0.6rem!important;color:var(--t2)!important;text-transform:uppercase!important;letter-spacing:.12em!important;font-weight:600!important;font-family:var(--fm)!important; white-space: normal; word-wrap: break-word;}
div[data-testid="stMetricValue"]{font-size:1.25rem!important;font-weight:800!important;color:#fff!important;line-height:1.1!important;margin-top:6px!important;font-family:var(--fd)!important;letter-spacing:-.03em!important; white-space: normal; word-wrap: break-word;}
div[data-testid="stMetricValue"]>div{color:#fff!important}
.stTabs [data-baseweb="tab-list"]{gap:0;background:transparent;border-bottom:1px solid rgba(201,168,76,0.08); flex-wrap: wrap;}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;border-bottom:2px solid transparent;border-radius:0;padding:12px 16px;color:var(--t3);font-weight:600;font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;font-family:var(--fm);transition:all .3s}
.stTabs [data-baseweb="tab"]:hover{color:var(--t1)}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--acc)!important;border-bottom-color:var(--acc)!important}
.stTabs [data-baseweb="tab-highlight"]{display:none}
.stDataFrame{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--b1)!important}
.stDownloadButton>button{background:rgba(201,168,76,.06)!important;color:var(--acc2)!important;border:1px solid var(--b2)!important;border-radius:10px!important;font-weight:600!important;padding:10px 24px!important;transition:all .3s!important}
.stDownloadButton>button:hover{background:rgba(201,168,76,.12)!important;box-shadow:0 4px 30px rgba(201,168,76,0.08)!important;transform:translateY(-1px)!important}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.04),rgba(201,168,76,.08),rgba(201,168,76,0.04),transparent)!important;margin:32px 0!important}
.vcard{background:linear-gradient(165deg,var(--s1),rgba(0,95,115,0.04));border:1px solid var(--b1);border-radius:16px;padding:26px 32px;margin-bottom:20px;position:relative;overflow:hidden}
.vcard::before{content:'';position:absolute;top:0;left:5%;right:5%;height:1px;background:linear-gradient(90deg,transparent,var(--acc2),transparent);opacity:.2}
.acard{background:var(--s1);border-radius:10px;padding:16px 20px;margin-bottom:8px;transition:transform .2s,box-shadow .2s}
.acard:hover{transform:translateX(3px);box-shadow:-3px 0 20px rgba(0,0,0,0.3)}
.pill{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.58rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-family:var(--fm)}
.p-ok{background:rgba(0,224,176,.06);color:var(--acc);border:1px solid rgba(0,224,176,.15)}
.p-w{background:rgba(212,168,67,.06);color:var(--amber);border:1px solid rgba(212,168,67,.15)}
.p-c{background:rgba(230,57,70,.06);color:var(--red);border:1px solid rgba(230,57,70,.15)}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--t3);border-radius:2px}
div[data-baseweb="select"] > div {background: var(--s1); border-color: var(--b1); color: #fff;}
</style>'''
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ICONS & UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def _u(s): return f"data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}"
ICONS={"VERIFIED":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#00e0b0" stroke-width="1" opacity=".2"><animate attributeName="r" values="12;13;12" dur="3s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#061a14" stroke="#00e0b0" stroke-width="1.2" filter="url(#g)"/><polyline points="10,14.5 12.8,17 18,10.5" fill="none" stroke="#00e0b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),"GHOST BUNKER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g2"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#e63946" stroke-width="1" stroke-dasharray="4 3"><animateTransform attributeName="transform" type="rotate" from="0 14 14" to="360 14 14" dur="8s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#1a0508" stroke="#e63946" stroke-width="1.2" filter="url(#g2)"/><g stroke="#e63946" stroke-width="2" stroke-linecap="round"><line x1="11" y1="11" x2="17" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line><line x1="17" y1="11" x2="11" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line></g></svg>'),"STAT OUTLIER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g4"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><rect x="4" y="4" width="20" height="20" rx="5" fill="none" stroke="#7b68ee" stroke-width="1.2" filter="url(#g4)"><animate attributeName="stroke-dasharray" values="0,80;80,0;0,80" dur="4s" repeatCount="indefinite"/></rect><circle cx="14" cy="14" r="4.5" fill="#0e0a1e" stroke="#7b68ee" stroke-width="1.2"/><circle cx="14" cy="14" r="1.8" fill="#7b68ee"><animate attributeName="r" values="1.8;2.8;1.8" dur="2s" repeatCount="indefinite"/></circle></svg>')}
SC={"VERIFIED":"#00e0b0","GHOST BUNKER":"#e63946","STAT OUTLIER":"#7b68ee"}

def _rgba(h,a): return f"rgba({int(h.lstrip('#')[0:2],16)},{int(h.lstrip('#')[2:4],16)},{int(h.lstrip('#')[4:6],16)},{a})"

def _sn(val):
    if pd.isna(val): return np.nan
    s = re.sub(r'[^\d.\-]', '', str(val).strip())
    try: return float(s) if s and s not in ('.','-','-.') else np.nan
    except: return np.nan
def _sn0(val): return 0.0 if np.isnan(_sn(val)) else _sn(val)

def _parse_dt(d_val,t_val):
    try:
        if isinstance(d_val, pd.Timestamp): d_str = d_val.strftime('%Y-%m-%d')
        elif pd.isna(d_val): return pd.NaT
        else:
            ds = str(d_val).strip()
            ds = re.sub(r'20224','2024',ds); ds = re.sub(r'20023','2023',ds)
            ds = re.sub(r'(\d+)\s+([A-Za-z]+)\.?\s+(\d{4})', lambda m:f"{m.group(3)}-{m.group(2)[:3]}-{m.group(1).zfill(2)}", ds)
            p = pd.to_datetime(ds, errors='coerce', format='mixed')
            if pd.isna(p): return pd.NaT
            d_str = p.strftime('%Y-%m-%d')
            
        if isinstance(t_val, pd.Timestamp): t_str = t_val.strftime('%H:%M')
        elif pd.isna(t_val): t_str = '00:00'
        else:
            tr = re.sub(r'[HhLlTtUuCc\s]', '', str(t_val).strip())
            m = re.match(r'^(\d{1,2}):(\d{2})', tr)
            if m: t_str = f"{m.group(1).zfill(2)}:{m.group(2)}"
            elif re.match(r'^\d{4}$', tr): t_str = f"{tr[:2]}:{tr[2:]}"
            elif re.match(r'^\d{3}$', tr): t_str = f"0{tr[0]}:{tr[1:]}"
            elif re.match(r'^\d{1,2}$', tr): t_str = f"{tr.zfill(2)}:00"
            else: t_str = '00:00'
        return pd.to_datetime(f"{d_str} {t_str}", errors='coerce')
    except: return pd.NaT

# ═══════════════════════════════════════════════════════════════════════════════
# AI CORE: ISOLATED THERMODYNAMICS (Trains strictly on Sea Legs)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def calculate_stochastic_variance(trip_df):
    # Initialize Safe Return Frame
    res_df = pd.DataFrame({
        'Stoch_Var':[0.0]*len(trip_df), 'SHAP_Base':[0.0]*len(trip_df), 'SHAP_Prop':[0.0]*len(trip_df), 
        'SHAP_Mass':[0.0]*len(trip_df), 'SHAP_Weath':[0.0]*len(trip_df), 'SHAP_Lag':[0.0]*len(trip_df),
        'Exp_Lower':[0.0]*len(trip_df), 'Exp_Upper':[0.0]*len(trip_df)
    }, index=trip_df.index)
    
    try:
        if not HAS_ML: return res_df
        
        ml = trip_df.copy()
        
        # Physics Engines
        ml['SOG'] = ml['Dist_NM'] / np.maximum(ml['Days']*24, 0.1)
        ml['Kin_Delta'] = (ml['Speed_kn'] - ml['SOG']).clip(-3.0, 3.0)
        ml['V3'] = ml['Speed_kn']**3
        ml['Cargo_MT'] = np.where(ml['Condition']=='LADEN', ml['CargoQty'], 0.0)
        ml['Froude_Proxy'] = (ml['Speed_kn']**2) * (ml['Cargo_MT'] / 10000.0)
        
        ml = ml.sort_values('Date_Start_TS')
        ml['Hull_EMA'] = ml['Kin_Delta'].ewm(span=10, adjust=False).mean()
        ml['Season'] = np.sin(2*np.pi*ml['Date_Start_TS'].dt.month.fillna(6)/12.0)
        
        # PURE PHYSICS ISOLATION: The AI ONLY trains on Sea Passages!
        sea_mask = (ml['Phase'] == 'SEA') & (ml['Daily_Burn'] > 1.0) & (ml['Days'] > 0.1) & (ml['Speed_kn'] > 2.0)
        if sea_mask.sum() < 2: return res_df # Require at least 2 valid ocean crossings

        # Lag calculation (only applied to valid sea legs)
        t_mod = XGBRegressor(n_estimators=20, max_depth=2, random_state=42)
        t_mod.fit(ml.loc[sea_mask, ['Speed_kn', 'Cargo_MT', 'Kin_Delta']], ml.loc[sea_mask, 'Daily_Burn'])
        ml['Lag'] = (ml['Daily_Burn'] - t_mod.predict(ml[['Speed_kn', 'Cargo_MT', 'Kin_Delta']])).shift(1).fillna(0).clip(-12, 12)

        features = ['Speed_kn', 'V3', 'Cargo_MT', 'Froude_Proxy', 'Kin_Delta', 'Hull_EMA', 'Season', 'Lag']
        ml[features] = ml[features].fillna(0.0)

        # Mean Conformal Model (Trained exclusively on pure sea data)
        model = XGBRegressor(n_estimators=100, max_depth=3, reg_lambda=5.0, learning_rate=0.05, random_state=42)
        model.fit(ml.loc[sea_mask, features], ml.loc[sea_mask, 'Daily_Burn'])
        
        # Variance Conformal Model (Trained exclusively on sea data residuals)
        residuals = np.abs(ml.loc[sea_mask, 'Daily_Burn'] - model.predict(ml.loc[sea_mask, features]))
        var_model = XGBRegressor(n_estimators=40, max_depth=2, reg_lambda=10.0, random_state=42)
        var_model.fit(ml.loc[sea_mask, features], residuals)
        
        # Predict ONLY for the Sea Legs
        sea_preds = model.predict(ml.loc[sea_mask, features])
        sea_vars = var_model.predict(ml.loc[sea_mask, features])
        sea_margins = np.maximum(sea_vars * 1.645, 0.5)
        
        # SHAP Execution
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(ml.loc[sea_mask, features])
        base = explainer.expected_value
        if isinstance(base, np.ndarray): base = base[0]

        # Re-inject the pristine AI data back into ONLY the Sea Rows of the dataframe
        res_df.loc[sea_mask, 'Stoch_Var'] = sea_margins.round(1)
        res_df.loc[sea_mask, 'SHAP_Base'] = base
        res_df.loc[sea_mask, 'SHAP_Prop'] = sv[:,0] + sv[:,1] + sv[:,3]
        res_df.loc[sea_mask, 'SHAP_Mass'] = sv[:,2]
        res_df.loc[sea_mask, 'SHAP_Weath'] = sv[:,4] + sv[:,5] + sv[:,6]
        res_df.loc[sea_mask, 'SHAP_Lag'] = sv[:,7]
        res_df.loc[sea_mask, 'Exp_Lower'] = sea_preds - sea_margins
        res_df.loc[sea_mask, 'Exp_Upper'] = sea_preds + sea_margins
        
        return res_df
        
    except Exception: 
        return res_df

# ═══════════════════════════════════════════════════════════════════════════════
# INGEST: THE TRI-STATE AD-TO-AD STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def process_file(uploaded_file):
    vn_raw = re.sub(r'\.[^.]+$', '', uploaded_file.name).strip()
    vname = re.sub(r'[_\-]+', ' ', vn_raw).upper()
    
    if uploaded_file.name.lower().endswith('.xlsx'):
        df_raw = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
    else:
        df_raw = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1', errors='replace')), header=None, on_bad_lines='skip')
        
    if df_raw.empty or len(df_raw) < 4: return pd.DataFrame(), vname, {}, []

    # 1. The Dual-Row Forward Fill Parser
    header_idx = 0
    cols_found = {}
    for i in range(min(60, len(df_raw))):
        vals = [str(x).upper() for x in df_raw.iloc[i].values if pd.notna(x)]
        if any(k in v for v in vals for k in ['DATE', 'DAY']) and any(k in v for v in vals for k in ['PORT', 'LOC']):
            header_idx = i
            top_header = df_raw.iloc[i].ffill()
            bottom_header = df_raw.iloc[i+1] if i+1 < len(df_raw) else pd.Series([np.nan]*len(df_raw.columns))
            
            for j in range(len(df_raw.columns)):
                c1 = str(top_header.iloc[j]).upper().strip() if pd.notna(top_header.iloc[j]) else ""
                c2 = str(bottom_header.iloc[j]).upper().strip() if pd.notna(bottom_header.iloc[j]) else ""
                c_combined = f"{c1} {c2}".strip()
                
                if 'VOY' in c_combined: cols_found['Voy'] = j
                elif 'PORT' in c_combined or 'LOC' in c_combined: cols_found['Port'] = j
                elif 'A/D' in c_combined or c_combined == 'AD' or 'STATUS' in c_combined: cols_found['AD'] = j
                elif 'SPEED' in c_combined: cols_found['Speed'] = j
                elif 'CARGO' in c_combined or 'QTY' in c_combined or 'QUANTITY' in c_combined:
                    if 'NAME' not in c_combined: cols_found['CargoQty'] = j
                elif 'DATE' in c_combined or 'DAY' in c_combined: cols_found['Date'] = j
                elif 'TIME' in c_combined and 'TOTAL' not in c_combined: cols_found['Time'] = j
                elif 'DIST' in c_combined and 'LEG' in c_combined: cols_found['DistLeg'] = j
                elif 'DIST' in c_combined and 'TOTAL' in c_combined: cols_found['TotalDist'] = j
                
                # Bunkers vs ROB
                elif 'BUNKER' in c1:
                    if 'FO' in c2 and 'MGO' not in c2: cols_found['Bunk_FO'] = j
                    elif 'MGO' in c2: cols_found['Bunk_MGO'] = j
                    elif 'MELO' in c2: cols_found['Bunk_MELO'] = j
                    elif 'HSCYLO' in c2 or 'HS CYLO' in c2: cols_found['Bunk_HSCYLO'] = j
                    elif 'LSCYLO' in c2 or 'LS CYLO' in c2: cols_found['Bunk_LSCYLO'] = j
                    elif 'GELO' in c2: cols_found['Bunk_GELO'] = j
                    elif 'CYLO' in c2: cols_found['Bunk_CYLO'] = j
                elif 'ROB' in c1:
                    if 'FO A' in c2: cols_found['FO_A'] = j
                    elif 'FO L' in c2: cols_found['FO_L'] = j
                    elif 'MGO A' in c2: cols_found['MGO_A'] = j
                    elif 'MGO L' in c2: cols_found['MGO_L'] = j
                    elif 'MELO' in c2: cols_found['MELO_R'] = j
                    elif 'HSCYLO' in c2 or 'HS CYLO' in c2: cols_found['HSCYLO_R'] = j
                    elif 'LSCYLO' in c2 or 'LS CYLO' in c2: cols_found['LSCYLO_R'] = j
                    elif 'GELO' in c2: cols_found['GELO_R'] = j
                    elif 'CYLO' in c2: cols_found['CYLO_R'] = j
            break
            
    df = df_raw.iloc[header_idx+1:].copy().reset_index(drop=True)
    
    for std_name, exc_idx in cols_found.items():
        if exc_idx < len(df.columns):
            if std_name in ['Voy', 'Port', 'AD', 'Date', 'Time']:
                df[std_name] = df.iloc[:, exc_idx]
            else:
                df[std_name] = df.iloc[:, exc_idx].apply(_sn).fillna(0.0)
                
    # 2. Crash-Proof Memory Initialization
    REQ_COLS = ['FO_A','FO_L','MGO_A','MGO_L','Bunk_FO','Bunk_MGO','Bunk_MELO','Bunk_HSCYLO','Bunk_LSCYLO','Bunk_GELO','MELO_R','HSCYLO_R','LSCYLO_R','GELO_R','CYLO_R','Speed','DistLeg','TotalDist','CargoQty','Voy','Port','AD']
    for req in REQ_COLS:
        if req not in df.columns:
            df[req] = 0.0 if req not in ['Voy','Port','AD'] else ''

    df['Datetime'] = df.apply(lambda r: _parse_dt(r.get('Date'), r.get('Time')), axis=1)
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    
    def _cad(v):
        v = str(v).strip().upper().replace(' ','')
        if v in ['D', 'DEP', 'SBE', 'FAOP']: return 'D'
        if v.startswith('A') and 'D' not in v: return 'A'
        if 'D' in v and 'A' not in v: return 'D'
        return v
    df['AD'] = df['AD'].apply(_cad)
    
    # 3. The Tri-State Event Extractor
    ad_events = df[df['AD'].isin(['A', 'D'])].copy()
    if len(ad_events) < 2: return pd.DataFrame(), vname, {}, []
    
    trips = []
    
    for i in range(len(ad_events)-1):
        r1 = ad_events.iloc[i]
        r2 = ad_events.iloc[i+1]
        
        idx1 = r1.name
        idx2 = r2.name
        
        # State Machine Core Logic
        if r1['AD'] == 'D' and r2['AD'] == 'A': phase = 'SEA'
        elif r1['AD'] == 'A' and r2['AD'] == 'D': phase = 'PORT'
        else: phase = 'SHIFT'
            
        days = (r2['Datetime'] - r1['Datetime']).total_seconds() / 86400.0
        if days < 0.05: continue # Ignore 1-hour micro shifts
        
        # Bunkering allocation (Bunkers log on departure day belong to the PORT stay)
        if phase == 'PORT':
            bfo = df.loc[idx1:idx2, 'Bunk_FO'].sum()
            bmgo = df.loc[idx1:idx2, 'Bunk_MGO'].sum()
            bmelo = df.loc[idx1:idx2, 'Bunk_MELO'].sum()
            bcylo = df.loc[idx1:idx2, 'Bunk_HSCYLO'].sum() + df.loc[idx1:idx2, 'Bunk_LSCYLO'].sum() + df.loc[idx1:idx2, 'Bunk_CYLO'].sum()
            bgelo = df.loc[idx1:idx2, 'Bunk_GELO'].sum()
        else: # SEA
            bfo = df.loc[idx1+1:idx2, 'Bunk_FO'].sum()
            bmgo = df.loc[idx1+1:idx2, 'Bunk_MGO'].sum()
            bmelo = df.loc[idx1+1:idx2, 'Bunk_MELO'].sum()
            bcylo = df.loc[idx1+1:idx2, 'Bunk_HSCYLO'].sum() + df.loc[idx1+1:idx2, 'Bunk_LSCYLO'].sum() + df.loc[idx1+1:idx2, 'Bunk_CYLO'].sum()
            bgelo = df.loc[idx1+1:idx2, 'Bunk_GELO'].sum()
            
        # Unified Mathematics via .get()
        hfo_c = (r1.get('FO_A', 0) - r2.get('FO_A', 0)) + bfo
        mgo_raw = (r1.get('MGO_A', 0) - r2.get('MGO_A', 0)) + bmgo
        mgo_c = max(0.0, mgo_raw)
        total_fuel = hfo_c + mgo_c
        
        burn_per_day = total_fuel / days if days > 0 else 0.0
        
        dist = df.loc[idx1+1:idx2, 'DistLeg'].sum()
        if dist <= 0 and phase == 'SEA':
            dist = max(0, r2.get('TotalDist',0) - r1.get('TotalDist',0))
            
        speed = df.loc[idx1+1:idx2, 'Speed'].replace(0, np.nan).mean()
        if pd.isna(speed): speed = dist / (days * 24.0) if days > 0 else 0.0
            
        melo_c = max(0, (r1.get('MELO_R',0) - r2.get('MELO_R',0)) + bmelo)
        cylo_r1 = r1.get('HSCYLO_R',0) + r1.get('LSCYLO_R',0) + r1.get('CYLO_R',0)
        cylo_r2 = r2.get('HSCYLO_R',0) + r2.get('LSCYLO_R',0) + r2.get('CYLO_R',0)
        cylo_c = max(0, (cylo_r1 - cylo_r2) + bcylo)
        gelo_c = max(0, (r1.get('GELO_R',0) - r2.get('GELO_R',0)) + bgelo)
        
        route = f"{str(r1.get('Port',''))[:15]} → {str(r2.get('Port',''))[:15]}" if phase == 'SEA' else f"In Port: {str(r1.get('Port',''))[:15]}"
            
        # The Quarantine Flags
        status = 'VERIFIED'
        flags = []
        if phase == 'PORT':
            if total_fuel < -5.0:
                status = 'GHOST BUNKER'
                flags.append(f"UNLOGGED BUNKER ~{abs(total_fuel):.0f}MT")
        elif phase == 'SEA':
            if total_fuel < -2.0:
                status = 'GHOST BUNKER' # Re-using icon for anomaly
                flags.append(f"NEGATIVE SEA BURN IMPOSSIBLE")
        
        cargo = str(r1.get('CargoName', '')).strip().upper()
        qty = _sn0(r1.get('CargoQty', 0))
        
        trips.append({
            'Indicator': ICONS.get(status, ICONS['VERIFIED']),
            'Timeline': f"{r1['Datetime'].strftime('%d %b %y')} → {r2['Datetime'].strftime('%d %b %y')}",
            'Date_Start_TS': r1['Datetime'],
            'Phase': phase,
            'Condition': 'LADEN' if qty > 100 else 'BALLAST',
            'CargoQty': qty,
            'Route': route,
            'Days': round(days, 2),
            'Dist_NM': round(dist, 0),
            'Speed_kn': round(speed, 1),
            'Fuel_MT': round(total_fuel, 1),
            'Daily_Burn': round(burn_per_day, 1),
            'MELO_L': round(melo_c, 0),
            'CYLO_L': round(cylo_c, 0),
            'GELO_L': round(gelo_c, 0),
            'Drift_MT': round(r1.get('FO_A', 0) - r1.get('FO_L', 0), 1),
            'Status': status,
            'Flags': ','.join(flags) if flags else ''
        })

    trip_df = pd.DataFrame(trips)
    
    # Statistical Outliers (Only applies to valid SEA legs)
    if len(trip_df) >= 4:
        for cond in ['LADEN', 'BALLAST']:
            ver = trip_df[(trip_df['Status'] == 'VERIFIED') & (trip_df['Phase'] == 'SEA') & (trip_df['Daily_Burn'] > 0) & (trip_df['Condition'] == cond)]
            if len(ver) >= 4:
                q1, q3 = ver['Daily_Burn'].quantile(0.25), ver['Daily_Burn'].quantile(0.75); iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 2.0*iqr, q3 + 2.0*iqr
                    mask = (trip_df['Status'] == 'VERIFIED') & (trip_df['Phase'] == 'SEA') & (trip_df['Condition'] == cond) & ((trip_df['Daily_Burn'] < lo) | (trip_df['Daily_Burn'] > hi))
                    trip_df.loc[mask, 'Status'] = 'STAT OUTLIER'; trip_df.loc[mask, 'Indicator'] = ICONS['STAT OUTLIER']

    # Inject the isolated AI Thermodynamics
    if not trip_df.empty:
        ai_df = calculate_stochastic_variance(trip_df)
        for col in ai_df.columns: trip_df[col] = ai_df[col]

    summary = {}
    if not trip_df.empty:
        n = len(trip_df)
        sea_burn = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Daily_Burn'] > 0)]['Daily_Burn']
        summary = {
            'integrity': round((trip_df['Status'] == 'VERIFIED').sum() / n * 100, 1),
            'total_fuel': round(trip_df['Fuel_MT'].sum(), 1),
            'total_hfo': round(trip_df['HFO_MT'].sum(), 1) if 'HFO_MT' in trip_df.columns else 0.0,
            'avg_sea_burn': round(sea_burn.mean(), 1) if not sea_burn.empty else 0.0,
            'total_nm': round(trip_df['Dist_NM'].sum(), 0),
            'total_melo': round(trip_df['MELO_L'].sum(), 0),
            'total_cylo': round(trip_df['CYLO_L'].sum(), 0),
            'total_days': round(trip_df['Days'].sum(), 1),
            'cycles': n,
            'anomalies': n - (trip_df['Status'] == 'VERIFIED').sum()
        }
        
    return trip_df, vname, summary, None

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS (Responsive & Non-Overlapping)
# ═══════════════════════════════════════════════════════════════════════════════
_BL=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',hovermode='x unified',hoverlabel=dict(bgcolor='#0c1219',bordercolor='rgba(201,168,76,0.12)',font=dict(family='Hanken Grotesk',color='#dce8f0',size=12)),font=dict(family='Hanken Grotesk',color='#4a6275',size=11),title_font=dict(family='Bricolage Grotesque',color='#ffffff',size=15),margin=dict(l=0,r=0,t=55,b=20))
_AX=dict(gridcolor='rgba(201,168,76,0.04)',zerolinecolor='rgba(201,168,76,0.06)',tickfont=dict(size=10))

def chart_fuel(df):
    sea = df[df['Phase'] == 'SEA'].copy()
    port = df[df['Phase'] == 'PORT'].copy()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    
    # Sea vs Port Fuel Bars
    if not sea.empty: fig.add_trace(go.Bar(x=sea['Timeline'], y=sea['Fuel_MT'], name='Sea Fuel', marker_color='rgba(0,224,176,0.15)', marker_line_color='#00e0b0', marker_line_width=1.5), row=1, col=1)
    if not port.empty: fig.add_trace(go.Bar(x=port['Timeline'], y=port['Fuel_MT'], name='Port/Idle Fuel', marker_color='rgba(123,104,238,0.15)', marker_line_color='#7b68ee', marker_line_width=1.5), row=1, col=1)
    
    # Sea MT/Day Line
    if not sea.empty: fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Daily_Burn'], name='Sea MT/day', mode='lines+markers', line=dict(color='#00e0b0', width=2, shape='spline')), row=1, col=1)
    
    # Speed Line
    if not sea.empty: fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Speed_kn'], name='Sea Speed', mode='lines+markers', line=dict(color='#c9a84c', width=2, shape='spline')), row=2, col=1)
    
    fig.update_layout(**_BL, title='Tri-State Fuel Consumption & Sea Speed Profile', barmode='group', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(tickangle=-45, automargin=True, **_AX); fig.update_yaxes(**_AX)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero"><div class="hero-left"><img src="data:image/svg+xml;base64,{_LOGO}" class="hero-logo" alt=""/><div><div class="hero-title">POSEIDON TITAN</div><div class="hero-sub">Fleet Consumables Intelligence Engine</div></div></div><div class="hero-badge"><span>KERNEL</span>&ensp;Tri-State AD-to-AD Ledge<br><span>PIPELINE</span>&ensp;Isolated Conformal Physics<br><span>BUILD</span>&ensp;v25.0 The Masterpiece</div></div>""",unsafe_allow_html=True)

uploaded_files=st.file_uploader('Upload vessel telemetry',accept_multiple_files=True,type=['xlsx','csv'],label_visibility='collapsed')

if not uploaded_files:
    st.markdown("""<div style="text-align:center;padding:100px 20px">
        <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:28px;opacity:.12">
            <circle cx="40" cy="40" r="36" fill="none" stroke="#c9a84c" stroke-width="0.8" stroke-dasharray="6 6"><animateTransform attributeName="transform" type="rotate" from="0 40 40" to="360 40 40" dur="30s" repeatCount="indefinite"/></circle>
            <circle cx="40" cy="40" r="24" fill="none" stroke="#00e0b0" stroke-width="0.6" stroke-dasharray="3 8"><animateTransform attributeName="transform" type="rotate" from="360 40 40" to="0 40 40" dur="20s" repeatCount="indefinite"/></circle>
            <path d="M40 14L40 66 M22 26Q40 36 58 26 M20 40Q40 50 60 40 M22 54Q40 64 58 54" fill="none" stroke="#00e0b0" stroke-width="1.5" stroke-linecap="round" opacity=".35"/></svg>
        <h2 style="color:#fff;font-family:'Bricolage Grotesque';font-weight:800;font-size:1.4rem;margin-bottom:8px;letter-spacing:-0.03em">Awaiting Telemetry</h2>
        <p style="color:#3a4d5e;font-size:.8rem;max-width:420px;margin:0 auto;line-height:1.7;font-family:'Hanken Grotesk'">Drop vessel noon-report files to execute the<br>AD-to-AD State Machine & Physics isolation.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

for f in uploaded_files:
    try:
        with st.spinner(f'Processing {f.name}...'):
            df, vname, summary, _ = process_file(f)
            
        if df.empty: st.warning(f'No valid events found in {f.name}.'); continue
        
        integrity = summary['integrity']
        ic = SC['VERIFIED'] if integrity >= 80 else (SC['LEDGER VARIANCE'] if integrity >= 50 else SC['GHOST BUNKER'])
        pc = 'p-ok' if integrity >= 80 else ('p-w' if integrity >= 50 else 'p-c')
        pt = 'NOMINAL' if integrity >= 80 else ('ATTENTION' if integrity >= 50 else 'CRITICAL')

        st.markdown(f"""<div class="vcard"><div style="display:flex;justify-content:space-between;align-items:center"><div><div style="font-family:var(--fd);font-weight:800;font-size:1.3rem;color:#fff;letter-spacing:-0.03em">{vname}</div><div style="font-family:var(--fm);font-size:.62rem;color:var(--t2);margin-top:5px;letter-spacing:0.04em">{summary['cycles']} LEGS&ensp;·&ensp;{summary['total_days']:.0f} DAYS&ensp;·&ensp;{int(summary['total_nm']):,} NM&ensp;·&ensp;{summary['total_fuel']:,.1f} MT</div><div style="font-family:var(--fm);font-size:.55rem;color:var(--t3);margin-top:3px;letter-spacing:0.04em">Tri-State Isolation Active</div></div><div style="text-align:right"><span class="pill {pc}">{pt}</span><div style="font-family:var(--fd);font-weight:800;font-size:1.6rem;color:{ic};margin-top:5px">{integrity:.0f}%</div><div style="font-family:var(--fm);font-size:.5rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em">Leg Verification</div></div></div></div>""",unsafe_allow_html=True)

        cols=st.columns(6)
        cols[0].metric('Total Fuel (MT)', f"{summary['total_fuel']:,.1f}")
        cols[1].metric('Total Days', f"{summary['total_days']:.1f}")
        cols[2].metric('Sea Burn (MT/d)', f"{summary['avg_sea_burn']:.1f}")
        cols[3].metric('Total MELO (L)', f"{int(summary['total_melo']):,}")
        cols[4].metric('Total CYLO (L)', f"{int(summary['total_cylo']):,}")
        cols[5].metric('Total Anomalies', f"{summary['anomalies']}")

        # --- THE 6-TAB ENTERPRISE SUITE ---
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['TRI-STATE MATRIX','FUEL & PHYSICS','LUBRICANTS','ANOMALY REVIEW','AI EXPLAINER (SHAP)', 'CONFORMAL BOUNDS'])

        with tab1:
            dcfg={'Indicator':st.column_config.ImageColumn(' ',width='small'),'Timeline':st.column_config.TextColumn('TIMELINE',width='medium'),'Phase':st.column_config.TextColumn('LEG',width='small'),'Condition':st.column_config.TextColumn('COND',width='small'),'Route':st.column_config.TextColumn('ROUTE',width='large'),'Days':st.column_config.NumberColumn('DAYS',format='%.2f'),'Dist_NM':st.column_config.NumberColumn('DIST',format='%d'),'Speed_kn':st.column_config.NumberColumn('SPD',format='%.1f'),'Fuel_MT':st.column_config.NumberColumn('TOTAL FUEL',format='%.1f'),'Daily_Burn':st.column_config.ProgressColumn('MT/DAY',format='%.1f',min_value=0,max_value=float(max(df['Daily_Burn'].max()*1.15,1))),'MELO_L':st.column_config.NumberColumn('MELO',format='%d'),'CYLO_L':st.column_config.NumberColumn('CYLO',format='%d'),'Stoch_Var':st.column_config.NumberColumn('AI VAR±',format='%.1f'),'Status':st.column_config.TextColumn('STATUS',width='medium'),'Flags':st.column_config.TextColumn('FLAGS',width='medium'),'Date_Start_TS':None,'SHAP_Base':None,'SHAP_Prop':None,'SHAP_Mass':None,'SHAP_Weath':None,'SHAP_Lag':None,'Exp_Lower':None,'Exp_Upper':None,'Drift_MT':None, 'CargoQty':None, 'GELO_L':None, 'HFO_MT':None, 'MGO_MT':None}
            st.dataframe(df,column_config=dcfg,hide_index=True,use_container_width=True,height=min(500,38+len(df)*35))
            buf=io.BytesIO(); exp=df.drop(columns=['Indicator','Date_Start_TS'],errors='ignore')
            with pd.ExcelWriter(buf,engine='openpyxl') as w: exp.to_excel(w,index=False,sheet_name='Audit')
            buf.seek(0)
            st.download_button('Export Tri-State Ledger',data=buf,file_name=f"{vname.replace(' ','_')}_LEDGER.xlsx",key=f"dl_{vname}")

        with tab2:
            st.plotly_chart(chart_fuel(df),use_container_width=True,config={'displayModeBar':False})

        with tab3:
            if df['MELO_L'].sum()+df['CYLO_L'].sum()+df['GELO_L'].sum()>0:
                st.plotly_chart(chart_lube(df),use_container_width=True,config={'displayModeBar':False})
            else: st.info('No lubricant consumption data detected.')

        with tab4:
            anomalies = df[df['Status'] != 'VERIFIED']
            if anomalies.empty:
                st.success("Zero anomalies detected. Fleet reporting is completely nominal.")
            else:
                for _,row in anomalies.iterrows():
                    s = row['Status']; sc = SC.get(s,'#fff'); ri = tuple(int(sc.lstrip('#')[i:i+2],16) for i in (0,2,4))
                    ai_v = row.get('Stoch_Var',0.0)
                    
                    desc = f"Unknown Exception in {row['Phase']} phase."
                    if row['Phase'] == 'PORT': desc = f"Mass Balance Error. Net fuel calculated at {row['Fuel_MT']:.1f} MT. Check port invoices for missing bunker receipts."
                    elif row['Phase'] == 'SEA': desc = f"Thermodynamic Anomaly. Reported burn of {row['Daily_Burn']:.1f} MT/d breached the Conformal limits (±{ai_v:.1f} MT) generated by the physics engine."
                    
                    st.markdown(f'<div class="acard" style="border:1px solid rgba({ri[0]},{ri[1]},{ri[2]},.12);border-left:3px solid {sc}"><div style="display:flex;justify-content:space-between;align-items:center"><div><span style="color:{sc};font-weight:700;font-size:.7rem;letter-spacing:.08em;font-family:var(--fm)">{s}</span><span style="color:var(--t3);font-size:.7rem;margin-left:10px;font-family:var(--fm)">{row["Timeline"]}</span></div><span style="color:var(--t2);font-size:.68rem;font-family:var(--fb)">{row["Route"]}</span></div><div style="color:var(--t2);font-size:.7rem;margin-top:8px;line-height:1.6;font-family:var(--fb)">{desc} <br><span style="color:var(--amber);font-family:var(--fm);font-size:0.6rem">[{row["Flags"]}]</span></div></div>',unsafe_allow_html=True)

        with tab5:
            st.markdown('<h3 style="color:#fff;font-family:var(--fd);font-size:1.2rem;margin-bottom:10px;margin-top:10px">Thermodynamic Logic Extraction (Sea Legs Only)</h3>', unsafe_allow_html=True)
            
            sea_df = df[df['Phase'] == 'SEA']
            shap_ran = sea_df['SHAP_Base'].abs().sum() > 0 if 'SHAP_Base' in sea_df.columns else False
            
            if not shap_ran:
                st.warning("⚠️ **AI EXPLAINABILITY OFFLINE:** The XGBoost engine requires at least 2 valid Sea Legs (>2.0 knots) to map the physical hydrodynamics. Port operations cannot be fed into the physics engine.")
            else:
                anomalies_s = sea_df[sea_df['Status'] != 'VERIFIED']
                if anomalies_s.empty:
                    st.success("No sea anomalies detected. View the AI's physics map for any valid ocean crossing below.")
                    options = sea_df['Timeline'].tolist()
                    sel = st.selectbox('Select Sea Passage', options, key=f'shap_{vname}')
                    tr = sea_df[sea_df['Timeline']==sel].iloc[0]
                else:
                    st.write("Select an anomalous Sea Passage to view the exact physics receipt vs reported fuel.")
                    options = anomalies_s['Timeline'].tolist()
                    sel = st.selectbox('Select Flagged Passage', options, key=f'shap_{vname}')
                    tr = anomalies_s[anomalies_s['Timeline']==sel].iloc[0]
                
                exp_burn = tr['SHAP_Base'] + tr['SHAP_Prop'] + tr['SHAP_Mass'] + tr['SHAP_Weath'] + tr['SHAP_Lag']
                
                fig_w = go.Figure(go.Waterfall(
                    name="SHAP", orientation="v",
                    measure=["absolute","relative","relative","relative","relative","total"],
                    x=["Fleet<br>Baseline", "Speed<br>& Froude", "Cargo<br>Mass", "Weather<br>& Drag", "Lag &<br>Bias", "Expected<br>Actual Burn"],
                    textposition="auto", insidetextanchor="middle",
                    text=[f"{tr['SHAP_Base']:.1f}",f"{tr['SHAP_Prop']:+.1f}",f"{tr['SHAP_Mass']:+.1f}",f"{tr['SHAP_Weath']:+.1f}",f"{tr['SHAP_Lag']:+.1f}",f"{exp_burn:.1f}"],
                    y=[tr['SHAP_Base'],tr['SHAP_Prop'],tr['SHAP_Mass'],tr['SHAP_Weath'],tr['SHAP_Lag'],0],
                    connector={"line":{"color":"rgba(201,168,76,0.15)"}},
                    decreasing={"marker":{"color":"#00e0b0"}}, increasing={"marker":{"color":"#e63946"}}, totals={"marker":{"color":"#7b68ee"}}
                ))
                fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color="#dce8f0",height=480,margin=dict(t=60,b=80,l=10,r=10),title=dict(text=f"Physics Audit: {tr['Route']} ({tr['Days']} Days)",font=dict(color='#ffffff',size=16,family='Bricolage Grotesque')),yaxis=dict(title='MT/Day',**_AX), xaxis=dict(automargin=True, tickangle=0, **_AX))
                st.plotly_chart(fig_w,use_container_width=True,config={'displayModeBar':False})
                
                st.info(f"**Forensic Translation:** The AI isolates the pure physics of this ocean crossing. Fleet baseline = **{tr['SHAP_Base']:.1f} MT/d**. Speed Effort: **{tr['SHAP_Prop']:+.1f}**. Mass: **{tr['SHAP_Mass']:+.1f}**. Environmental Drag: **{tr['SHAP_Weath']:+.1f}**. Memory Bias: **{tr['SHAP_Lag']:+.1f}**. \n\nExpected Mathematical Burn: **{exp_burn:.1f} MT/d** vs Physically Reported: **{tr['Daily_Burn']:.1f} MT/d**.")

        with tab6:
            sea_df = df[df['Phase'] == 'SEA']
            if 'Exp_Lower' in sea_df.columns and sea_df['Exp_Lower'].abs().sum() > 0:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=sea_df['Timeline'].tolist() + sea_df['Timeline'].tolist()[::-1],
                                         y=sea_df['Exp_Upper'].tolist() + sea_df['Exp_Lower'].tolist()[::-1],
                                         fill='toself', fillcolor='rgba(123,104,238,0.15)', line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip", name='90% Conformal Interval'))
                fig_c.add_trace(go.Scatter(x=sea_df['Timeline'], y=(sea_df['Exp_Lower']+sea_df['Exp_Upper'])/2, name="Expected Mean", line=dict(color="#7b68ee", width=2, dash='dot')))
                fig_c.add_trace(go.Scatter(x=sea_df['Timeline'], y=sea_df['Daily_Burn'], name="Actual Burn (FO_A)", mode='lines+markers', line=dict(color="#00e0b0", width=2), marker=dict(size=6, color="#fff")))
                
                fig_c.update_layout(**_BL, title='Conformal Bounds (Sea Legs Only)', height=500, yaxis=dict(title='MT/day', **_AX), xaxis=dict(tickangle=-45, automargin=True, **_AX))
                st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar':False})
                st.caption("The purple envelope represents the absolute physical limits of the vessel based on XGBoost variance. Port operations are strictly excluded from this visualization to guarantee accuracy.")
            else:
                st.info("Insufficient sea data to generate Conformal Prediction Bounds.")

        st.divider()
    except Exception:
        st.error(f'System Failure on file: {f.name}')
        with st.expander('View Traceback'): st.code(traceback.format_exc())
