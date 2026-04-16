import streamlit as st
import pandas as pd
import numpy as np
import re, io, math, traceback, base64, warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# TITAN CORE: v26.0 ZERO-TOLERANCE FORENSIC AUDITOR
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

# ANTI-OVERLAP CSS & GLASS BOX UI
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
[data-testid="stFileUploader"]{background:var(--s1)!important;border:1px dashed var(--b2)!important;border-radius:var(--r)!important;padding:24px!important;transition:all .3s}
[data-testid="stFileUploader"]:hover{border-color:var(--acc2)!important;box-shadow:0 0 20px rgba(201,168,76,0.08)}
div[data-testid="stMetric"]{background:linear-gradient(180deg,var(--s1),var(--s2))!important;border:1px solid var(--b1)!important;border-radius:var(--r);padding:18px 15px!important;position:relative;overflow:hidden;}
div[data-testid="stMetricLabel"]{font-size:.6rem!important;color:var(--t2)!important;text-transform:uppercase!important;letter-spacing:.12em!important;font-weight:600!important;font-family:var(--fm)!important; white-space: normal; word-wrap: break-word;}
div[data-testid="stMetricValue"]{font-size:1.4rem!important;font-weight:800!important;color:#fff!important;line-height:1.1!important;margin-top:6px!important;font-family:var(--fd)!important; white-space: normal; word-wrap: break-word;}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;border-bottom:2px solid transparent;border-radius:0;padding:12px 18px;color:var(--t3);font-weight:600;font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;font-family:var(--fm);transition:all .3s}
.stTabs [aria-selected="true"]{color:var(--acc)!important;border-bottom-color:var(--acc)!important}
.stDataFrame{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--b1)!important}
.acard{background:var(--s1);border-radius:10px;padding:16px 20px;margin-bottom:8px;border-left:3px solid var(--red); transition:transform .2s}
.acard:hover{transform:translateX(3px)}
.pill{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.55rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-family:var(--fm)}
.p-ok{background:rgba(0,224,176,.06);color:var(--acc);border:1px solid rgba(0,224,176,.15)}
.p-w{background:rgba(212,168,67,.06);color:var(--amber);border:1px solid rgba(212,168,67,.15)}
.p-c{background:rgba(230,57,70,.06);color:var(--red);border:1px solid rgba(230,57,70,.15)}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--t3);border-radius:2px}
</style>'''
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STRICT PARSING UTILS (No Hallucinated Data)
# ═══════════════════════════════════════════════════════════════════════════════
def _u(s): return f"data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}"
ICONS={"VERIFIED":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#00e0b0" stroke-width="1" opacity=".2"><animate attributeName="r" values="12;13;12" dur="3s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#061a14" stroke="#00e0b0" stroke-width="1.2" filter="url(#g)"/><polyline points="10,14.5 12.8,17 18,10.5" fill="none" stroke="#00e0b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),"GHOST BUNKER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g2"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#e63946" stroke-width="1" stroke-dasharray="4 3"><animateTransform attributeName="transform" type="rotate" from="0 14 14" to="360 14 14" dur="8s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#1a0508" stroke="#e63946" stroke-width="1.2" filter="url(#g2)"/><g stroke="#e63946" stroke-width="2" stroke-linecap="round"><line x1="11" y1="11" x2="17" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line><line x1="17" y1="11" x2="11" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line></g></svg>'),"STAT OUTLIER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g4"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><rect x="4" y="4" width="20" height="20" rx="5" fill="none" stroke="#7b68ee" stroke-width="1.2" filter="url(#g4)"><animate attributeName="stroke-dasharray" values="0,80;80,0;0,80" dur="4s" repeatCount="indefinite"/></rect><circle cx="14" cy="14" r="4.5" fill="#0e0a1e" stroke="#7b68ee" stroke-width="1.2"/><circle cx="14" cy="14" r="1.8" fill="#7b68ee"><animate attributeName="r" values="1.8;2.8;1.8" dur="2s" repeatCount="indefinite"/></circle></svg>')}
SC={"VERIFIED":"#00e0b0","GHOST BUNKER":"#e63946","STAT OUTLIER":"#7b68ee"}

def _sn(val):
    # Strict Null: Returns np.nan if blank. Never returns 0.0 for missing data.
    if pd.isna(val): return np.nan
    s = re.sub(r'[^\d.\-]', '', str(val).strip())
    try: return float(s) if s and s not in ('.','-','-.') else np.nan
    except: return np.nan

def _sn0(val):
    # Zero Null: Only used for quantities like Cargo or Bunkers where blank = 0
    v = _sn(val)
    return 0.0 if np.isnan(v) else v

def _parse_dt(d_val, t_val):
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
# AI ENGINE: ISOLATED KINEMATICS (Sea Legs Only)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_ai_physics(trip_df):
    res_df = pd.DataFrame({
        'AI_Exp':[np.nan]*len(trip_df), 'Stoch_Var':[np.nan]*len(trip_df), 
        'SHAP_Base':[np.nan]*len(trip_df), 'SHAP_Prop':[np.nan]*len(trip_df), 
        'SHAP_Mass':[np.nan]*len(trip_df), 'SHAP_Weath':[np.nan]*len(trip_df), 
        'Exp_Lower':[np.nan]*len(trip_df), 'Exp_Upper':[np.nan]*len(trip_df)
    }, index=trip_df.index)
    
    if not HAS_ML: return res_df
    
    try:
        # STRICT ISOLATION: The AI physically cannot see Port Legs or Quarantined Legs
        sea_mask = (trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Speed_kn'] > 2.0)
        if sea_mask.sum() < 2: return res_df
        
        ml = trip_df.loc[sea_mask].copy()
        
        # PURE KINEMATICS
        ml['SOG'] = ml['Dist_NM'] / np.maximum(ml['Days']*24, 0.1)
        ml['Kin_Delta'] = (ml['Speed_kn'] - ml['SOG']).clip(-3.0, 3.0)
        ml['V3'] = ml['Speed_kn']**3
        ml['Cargo_MT'] = np.where(ml['Condition']=='LADEN', ml['CargoQty'], 0.0)
        ml['Froude_Proxy'] = (ml['Speed_kn']**3) * (ml['Cargo_MT'] / 10000.0) # V^3 * Mass interaction
        
        ml = ml.sort_values('Date_Start_TS')
        ml['Hull_EMA'] = ml['Kin_Delta'].ewm(span=10, adjust=False).mean()
        ml['Season'] = np.sin(2*np.pi*ml['Date_Start_TS'].dt.month.fillna(6)/12.0)
        
        features = ['Speed_kn', 'V3', 'Cargo_MT', 'Froude_Proxy', 'Kin_Delta', 'Hull_EMA', 'Season']
        ml[features] = ml[features].fillna(0.0)

        # MEAN MODEL
        model = XGBRegressor(n_estimators=100, max_depth=3, reg_lambda=5.0, learning_rate=0.05, random_state=42)
        model.fit(ml[features], ml['Phys_Burn'])
        
        # VARIANCE MODEL
        residuals = np.abs(ml['Phys_Burn'] - model.predict(ml[features]))
        var_model = XGBRegressor(n_estimators=40, max_depth=2, reg_lambda=10.0, random_state=42)
        var_model.fit(ml[features], residuals)
        
        preds = model.predict(ml[features])
        margins = np.maximum(var_model.predict(ml[features]) * 1.645, 0.5) 
        
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(ml[features])
        base = explainer.expected_value
        if isinstance(base, np.ndarray): base = base[0]

        # Inject back only into verified Sea Legs
        res_df.loc[sea_mask, 'AI_Exp'] = preds.round(1)
        res_df.loc[sea_mask, 'Stoch_Var'] = margins.round(1)
        res_df.loc[sea_mask, 'SHAP_Base'] = base
        res_df.loc[sea_mask, 'SHAP_Prop'] = sv[:,0] + sv[:,1] + sv[:,3] # Speed + V3 + Froude
        res_df.loc[sea_mask, 'SHAP_Mass'] = sv[:,2]
        res_df.loc[sea_mask, 'SHAP_Weath'] = sv[:,4] + sv[:,5] + sv[:,6] # Kin_Delta + Hull_EMA + Season
        res_df.loc[sea_mask, 'Exp_Lower'] = preds - margins
        res_df.loc[sea_mask, 'Exp_Upper'] = preds + margins
        
        return res_df
    except Exception: return res_df

# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION: THE FORENSIC AD-TO-AD STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def master_ingest(uploaded_file):
    vn_raw = re.sub(r'\.[^.]+$', '', uploaded_file.name).strip()
    vname = re.sub(r'[_\-]+', ' ', vn_raw).upper()
    
    if uploaded_file.name.lower().endswith('.xlsx'): df_raw = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
    else: df_raw = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1', errors='replace')), header=None, on_bad_lines='skip')
        
    if df_raw.empty or len(df_raw) < 4: return pd.DataFrame(), vname, {}

    # SEMANTIC HEADER PARSER (Explicit Mapping)
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
                
                elif 'BUNKER' in c1 or 'RECEIV' in c1:
                    if 'FO' in c2 and 'MGO' not in c2: cols_found['Bunk_FO'] = j
                    elif 'MGO' in c2: cols_found['Bunk_MGO'] = j
                elif 'ROB' in c1:
                    if 'FO A' in c2: cols_found['FO_A'] = j
                    elif 'FO L' in c2: cols_found['FO_L'] = j
            break
            
    df = df_raw.iloc[header_idx+1:].copy().reset_index(drop=True)
    
    # Safe Extraction
    for std_name, exc_idx in cols_found.items():
        if exc_idx < len(df.columns):
            if std_name in ['Voy', 'Port', 'AD', 'Date', 'Time']: df[std_name] = df.iloc[:, exc_idx]
            else: df[std_name] = df.iloc[:, exc_idx] # Keep raw for strict parsing

    # Guarantee Dictionary Structure
    REQ_COLS = ['FO_A','FO_L','Bunk_FO','Bunk_MGO','Speed','DistLeg','TotalDist','CargoQty','Voy','Port','AD']
    for req in REQ_COLS:
        if req not in df.columns: df[req] = np.nan if req in ['FO_A', 'FO_L'] else (0.0 if req not in ['Voy','Port','AD'] else '')

    df['Datetime'] = df.apply(lambda r: _parse_dt(r.get('Date'), r.get('Time')), axis=1)
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    
    def _cad(v):
        v = str(v).strip().upper().replace(' ','')
        if v in ['D', 'DEP', 'SBE', 'FAOP']: return 'D'
        if v.startswith('A') and 'D' not in v: return 'A'
        return v
    df['AD'] = df['AD'].apply(_cad)
    
    ad_events = df[df['AD'].isin(['A', 'D'])].copy()
    if len(ad_events) < 2: return pd.DataFrame(), vname, {}
    
    trips = []
    
    # CHAIN OF CUSTODY VALIDATION
    for i in range(len(ad_events)-1):
        r1 = ad_events.iloc[i]
        r2 = ad_events.iloc[i+1]
        idx1, idx2 = r1.name, r2.name
        
        status = 'VERIFIED'
        flags = []
        phys_burn, log_burn, drift, days = np.nan, np.nan, np.nan, 0.0
        
        # 1. State Verification
        if r1['AD'] == 'D' and r2['AD'] == 'A': phase = 'SEA'
        elif r1['AD'] == 'A' and r2['AD'] == 'D': phase = 'PORT'
        else: 
            phase = 'QUARANTINE_SEQ'
            status = 'CHAIN BREAK'
            flags.append(f"Invalid sequence: {r1['AD']} → {r2['AD']}")
            
        # 2. Time Verification
        if status == 'VERIFIED':
            days = (r2['Datetime'] - r1['Datetime']).total_seconds() / 86400.0
            if days <= 0:
                status = 'QUARANTINE_TIME'
                flags.append("Negative/Zero Time Delta")
                
        # 3. Physical ROB Verification (No Guessing allowed)
        if status == 'VERIFIED':
            start_rob = _sn(r1.get('FO_A'))
            end_rob = _sn(r2.get('FO_A'))
            if pd.isna(start_rob) or pd.isna(end_rob):
                status = 'QUARANTINE_ROB'
                flags.append("Missing Physical Tank Sounding (FO_A)")
        
        # Window Operations
        window = df.loc[idx1+1:idx2]
        if phase == 'PORT': bfo = _sn0(df.loc[idx1:idx2, 'Bunk_FO'].sum())
        else: bfo = _sn0(window['Bunk_FO'].sum())
        
        dist = _sn0(window['DistLeg'].sum()) if 'DistLeg' in window.columns else 0.0
        if dist <= 0 and phase == 'SEA': dist = max(0, _sn0(r2.get('TotalDist')) - _sn0(r1.get('TotalDist')))
        
        speed = window['Speed'].replace(0, np.nan).mean() if not window['Speed'].empty else np.nan
        if pd.isna(speed): speed = dist / (days * 24.0) if days > 0 else 0.0
        
        # 4. Deterministic Arithmetic (Glass Box)
        if status == 'VERIFIED':
            phys_burn = (start_rob - end_rob) + bfo
            
            # Logged verification
            log_start = _sn(r1.get('FO_L')) if not pd.isna(_sn(r1.get('FO_L'))) else start_rob
            log_end = _sn(r2.get('FO_L')) if not pd.isna(_sn(r2.get('FO_L'))) else end_rob
            log_burn = (log_start - log_end) + bfo
            
            drift = phys_burn - log_burn
            
            if phys_burn < -2.0:
                status = 'GHOST BUNKER'
                flags.append("Mass Balance Imbalance: End ROB > Start + Bunkers")
                
        route = f"{str(r1.get('Port',''))[:15]} → {str(r2.get('Port',''))[:15]}" if phase == 'SEA' else f"Port Idle: {str(r1.get('Port',''))[:15]}"
        qty = _sn0(r1.get('CargoQty', 0))
        
        trips.append({
            'Indicator': ICONS.get(status, ICONS['VERIFIED']) if 'QUARANTINE' not in status else '⛔',
            'Timeline': f"{r1['Datetime'].strftime('%d %b %y')} → {r2['Datetime'].strftime('%d %b %y')}",
            'Date_Start_TS': r1['Datetime'],
            'Phase': phase,
            'Condition': 'LADEN' if qty > 100 else 'BALLAST',
            'CargoQty': qty,
            'Route': route,
            'Days': round(days, 2),
            'Dist_NM': round(dist, 0),
            'Speed_kn': round(speed, 1),
            'FO_A_Start': start_rob if status == 'VERIFIED' else np.nan,
            'Bunk_FO': bfo,
            'FO_A_End': end_rob if status == 'VERIFIED' else np.nan,
            'Phys_Burn': round(phys_burn, 1),
            'Log_Burn': round(log_burn, 1),
            'Drift_MT': round(drift, 1),
            'Daily_Burn': round(phys_burn/days, 1) if days > 0 and not pd.isna(phys_burn) else np.nan,
            'Status': status,
            'Flags': ', '.join(flags) if flags else ''
        })

    trip_df = pd.DataFrame(trips)
    
    # Inject isolated AI bounds
    if not trip_df.empty:
        ai_df = compute_ai_physics(trip_df)
        for col in ai_df.columns: trip_df[col] = ai_df[col]

    summary = {}
    if not trip_df.empty:
        verified_sea = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Phys_Burn'] > 0)]['Phys_Burn']
        verified_sea_days = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Phys_Burn'] > 0)]['Days']
        avg_sea = verified_sea.sum() / verified_sea_days.sum() if verified_sea_days.sum() > 0 else 0.0
        
        quarantined = len(trip_df[trip_df['Status'].str.contains('QUARANTINE')])
        
        summary = {
            'integrity': round((len(trip_df) - quarantined) / len(trip_df) * 100, 1),
            'total_fuel': round(trip_df['Phys_Burn'].sum(skipna=True), 1),
            'avg_sea_burn': round(avg_sea, 1),
            'total_nm': round(trip_df['Dist_NM'].sum(), 0),
            'total_days': round(trip_df['Days'].sum(), 1),
            'cycles': len(trip_df),
            'quarantined': quarantined,
            'anomalies': len(trip_df[trip_df['Status'] == 'GHOST BUNKER'])
        }
        
    return trip_df, vname, summary

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI (Epistemological Segregation)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero"><div class="hero-left"><img src="data:image/svg+xml;base64,{_LOGO}" class="hero-logo" alt=""/><div><div class="hero-title">POSEIDON TITAN</div><div class="hero-sub">Forensic Accounting & Intelligence Engine</div></div></div><div class="hero-badge"><span>KERNEL</span>&ensp;Zero-Tolerance State Machine<br><span>PIPELINE</span>&ensp;Epistemological AI Isolation<br><span>BUILD</span>&ensp;v26.0 Masterpiece</div></div>""",unsafe_allow_html=True)

uploaded_files=st.file_uploader('Upload vessel telemetry',accept_multiple_files=True,type=['xlsx','csv'],label_visibility='collapsed')

if not uploaded_files:
    st.markdown("""<div style="text-align:center;padding:100px 20px">
        <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:28px;opacity:.12">
            <circle cx="40" cy="40" r="36" fill="none" stroke="#c9a84c" stroke-width="0.8" stroke-dasharray="6 6"><animateTransform attributeName="transform" type="rotate" from="0 40 40" to="360 40 40" dur="30s" repeatCount="indefinite"/></circle>
            <circle cx="40" cy="40" r="24" fill="none" stroke="#00e0b0" stroke-width="0.6" stroke-dasharray="3 8"><animateTransform attributeName="transform" type="rotate" from="360 40 40" to="0 40 40" dur="20s" repeatCount="indefinite"/></circle>
            <path d="M40 14L40 66 M22 26Q40 36 58 26 M20 40Q40 50 60 40 M22 54Q40 64 58 54" fill="none" stroke="#00e0b0" stroke-width="1.5" stroke-linecap="round" opacity=".35"/></svg>
        <h2 style="color:#fff;font-family:'Bricolage Grotesque';font-weight:800;font-size:1.4rem;margin-bottom:8px;letter-spacing:-0.03em">Awaiting Telemetry</h2>
        <p style="color:#3a4d5e;font-size:.8rem;max-width:420px;margin:0 auto;line-height:1.7;font-family:'Hanken Grotesk'">Drop vessel noon-report files to execute the<br>Zero-Tolerance State Machine & Physics isolation.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

for f in uploaded_files:
    try:
        with st.spinner(f'Auditing {f.name}...'):
            df, vname, summary = master_ingest(f)
            
        if df.empty: st.warning(f'No valid events found in {f.name}. Check the template.'); continue
        
        integrity = summary['integrity']
        ic = SC['VERIFIED'] if integrity >= 80 else (SC['LEDGER VARIANCE'] if integrity >= 50 else SC['GHOST BUNKER'])
        pc = 'p-ok' if integrity >= 80 else ('p-w' if integrity >= 50 else 'p-c')
        pt = 'NOMINAL' if integrity >= 80 else ('ATTENTION' if integrity >= 50 else 'CRITICAL')

        st.markdown(f"""<div class="vcard"><div style="display:flex;justify-content:space-between;align-items:center"><div><div style="font-family:var(--fd);font-weight:800;font-size:1.3rem;color:#fff;letter-spacing:-0.03em">{vname}</div><div style="font-family:var(--fm);font-size:.62rem;color:var(--t2);margin-top:5px;letter-spacing:0.04em">{summary['cycles']} LEGS&ensp;·&ensp;{summary['total_days']:.0f} DAYS&ensp;·&ensp;{int(summary['total_nm']):,} NM</div><div style="font-family:var(--fm);font-size:.55rem;color:var(--t3);margin-top:3px;letter-spacing:0.04em">Deterministic Core Active</div></div><div style="text-align:right"><span class="pill {pc}">{pt}</span><div style="font-family:var(--fd);font-weight:800;font-size:1.6rem;color:{ic};margin-top:5px">{integrity:.0f}%</div><div style="font-family:var(--fm);font-size:.5rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em">Audit Integrity</div></div></div></div>""",unsafe_allow_html=True)

        cols=st.columns(5)
        cols[0].metric('Verified Fuel (MT)', f"{summary['total_fuel']:,.1f}")
        cols[1].metric('Avg Sea Burn (MT/d)', f"{summary['avg_sea_burn']:.1f}")
        cols[2].metric('Total Distance (NM)', f"{int(summary['total_nm']):,}")
        cols[3].metric('Mass Anomalies', f"{summary['anomalies']}")
        cols[4].metric('Quarantined Legs', f"{summary['quarantined']}")

        # EPISTEMOLOGICAL TABS
        tab1,tab2,tab3,tab4 = st.tabs(['IMMUTABLE LEDGER (Deterministic)','DIGITAL TWIN (AI Forensics)', 'CONFORMAL BOUNDS', 'QUARANTINE LOG'])

        with tab1:
            st.markdown('<span style="font-family:var(--fm); font-size:0.7rem; color:var(--acc)">[START ROB] + [BUNKERS] - [END ROB] = [PHYSICAL BURN]</span>', unsafe_allow_html=True)
            dcfg={'Indicator':st.column_config.ImageColumn(' ',width='small'),'Timeline':st.column_config.TextColumn('TIMELINE',width='medium'),'Phase':st.column_config.TextColumn('LEG',width='small'),'Days':st.column_config.NumberColumn('DAYS',format='%.2f'),'Speed_kn':st.column_config.NumberColumn('SPD',format='%.1f'),'FO_A_Start':st.column_config.NumberColumn('START ROB',format='%.1f'),'Bunk_FO':st.column_config.NumberColumn('+ BUNKERS',format='%.1f'),'FO_A_End':st.column_config.NumberColumn('- END ROB',format='%.1f'),'Phys_Burn':st.column_config.NumberColumn('= PHYS BURN',format='%.1f'),'Log_Burn':st.column_config.NumberColumn('LOG BURN',format='%.1f'),'Drift_MT':st.column_config.NumberColumn('DRIFT ±',format='%.1f'),'Daily_Burn':st.column_config.NumberColumn('MT/DAY',format='%.1f'),'Status':st.column_config.TextColumn('STATUS',width='medium'),'Route':None,'CargoQty':None,'Condition':None,'Date_Start_TS':None,'Flags':None,'AI_Exp':None,'Stoch_Var':None,'SHAP_Base':None,'SHAP_Prop':None,'SHAP_Mass':None,'SHAP_Weath':None,'Exp_Lower':None,'Exp_Upper':None,'Dist_NM':None}
            st.dataframe(df,column_config=dcfg,hide_index=True,use_container_width=True,height=min(500,38+len(df)*35))

        with tab2:
            st.markdown('<h3 style="color:#fff;font-family:var(--fd);font-size:1.2rem;margin-bottom:10px;margin-top:10px">Thermodynamic Explanation (Sea Legs Only)</h3>', unsafe_allow_html=True)
            sea_df = df[(df['Phase'] == 'SEA') & (df['Status'] == 'VERIFIED')]
            shap_ran = sea_df['SHAP_Base'].abs().sum() > 0 if 'SHAP_Base' in sea_df.columns else False
            
            if not shap_ran:
                st.warning("⚠️ **AI EXPLAINABILITY OFFLINE:** Insufficient valid Sea Legs (>2.0 knots) to train the hydrodynamic model.")
            else:
                options = sea_df['Timeline'].tolist()
                sel = st.selectbox('Select Verified Sea Passage', options, key=f'shap_{vname}')
                tr = sea_df[sea_df['Timeline']==sel].iloc[0]
                
                exp_burn = tr['SHAP_Base'] + tr['SHAP_Prop'] + tr['SHAP_Mass'] + tr['SHAP_Weath']
                
                fig_w = go.Figure(go.Waterfall(
                    name="SHAP", orientation="v",
                    measure=["absolute","relative","relative","relative","total"],
                    x=["Fleet<br>Baseline", "Speed<br>& Froude", "Cargo<br>Mass", "Weather<br>& Drag", "Expected<br>Burn"],
                    textposition="auto", insidetextanchor="middle",
                    text=[f"{tr['SHAP_Base']:.1f}",f"{tr['SHAP_Prop']:+.1f}",f"{tr['SHAP_Mass']:+.1f}",f"{tr['SHAP_Weath']:+.1f}",f"{exp_burn:.1f}"],
                    y=[tr['SHAP_Base'],tr['SHAP_Prop'],tr['SHAP_Mass'],tr['SHAP_Weath'],0],
                    connector={"line":{"color":"rgba(201,168,76,0.15)"}},
                    decreasing={"marker":{"color":"#00e0b0"}}, increasing={"marker":{"color":"#e63946"}}, totals={"marker":{"color":"#7b68ee"}}
                ))
                fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color="#dce8f0",height=480,margin=dict(t=60,b=80,l=10,r=10),title=dict(text=f"Physics Audit: {tr['Route']} ({tr['Days']} Days at {tr['Speed_kn']}kn)",font=dict(color='#ffffff',size=16,family='Bricolage Grotesque')),yaxis=dict(title='MT/Day', gridcolor='rgba(201,168,76,0.04)'), xaxis=dict(automargin=True, tickangle=0))
                st.plotly_chart(fig_w,use_container_width=True,config={'displayModeBar':False})
                
                st.info(f"**Forensic Context:** The AI isolated the physics of this ocean crossing. Expected Mathematical Burn: **{exp_burn:.1f} MT/d** vs Physically Audited Burn: **{tr['Daily_Burn']:.1f} MT/d**.")

        with tab3:
            sea_df = df[(df['Phase'] == 'SEA') & (df['Status'] == 'VERIFIED')]
            if 'Exp_Lower' in sea_df.columns and sea_df['Exp_Lower'].abs().sum() > 0:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=sea_df['Timeline'].tolist() + sea_df['Timeline'].tolist()[::-1],
                                         y=sea_df['Exp_Upper'].tolist() + sea_df['Exp_Lower'].tolist()[::-1],
                                         fill='toself', fillcolor='rgba(123,104,238,0.15)', line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip", name='90% Conformal Interval'))
                fig_c.add_trace(go.Scatter(x=sea_df['Timeline'], y=sea_df['AI_Exp'], name="Expected Mean", line=dict(color="#7b68ee", width=2, dash='dot')))
                fig_c.add_trace(go.Scatter(x=sea_df['Timeline'], y=sea_df['Daily_Burn'], name="Audited Burn", mode='lines+markers', line=dict(color="#00e0b0", width=2), marker=dict(size=6, color="#fff")))
                
                fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color="#dce8f0", title='Conformal Propulsion Bounds (Verified Sea Legs Only)', height=500, yaxis=dict(title='MT/day', gridcolor='rgba(201,168,76,0.04)'), xaxis=dict(tickangle=-45, automargin=True, gridcolor='rgba(201,168,76,0.04)'))
                st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar':False})
            else:
                st.info("Insufficient verified sea data to generate prediction bounds.")

        with tab4:
            quarantined = df[df['Status'].str.contains('QUARANTINE|GHOST')]
            if quarantined.empty:
                st.success("Zero structural anomalies detected. All chronological chains and tank soundings are intact.")
            else:
                for _,row in quarantined.iterrows():
                    s = row['Status']; c = '#e63946'
                    desc = f"Fatal Audit Exception: {row['Flags']}"
                    st.markdown(f'<div class="acard"><div style="display:flex;justify-content:space-between;align-items:center"><div><span style="color:{c};font-weight:700;font-size:.7rem;letter-spacing:.08em;font-family:var(--fm)">{s}</span><span style="color:var(--t3);font-size:.7rem;margin-left:10px;font-family:var(--fm)">{row["Timeline"]}</span></div><span style="color:var(--t2);font-size:.68rem;font-family:var(--fb)">{row["Route"]}</span></div><div style="color:var(--t2);font-size:.7rem;margin-top:8px;line-height:1.6;font-family:var(--fb)">{desc}</div></div>',unsafe_allow_html=True)

        st.divider()
    except Exception:
        st.error(f'System Failure on file: {f.name}')
        with st.expander('View Traceback'): st.code(traceback.format_exc())
