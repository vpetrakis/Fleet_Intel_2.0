import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import math
import traceback
import base64
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from xgboost import XGBRegressor
    import shap
    HAS_ML = True
except ImportError:
    HAS_ML = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: CONFIGURATION & SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
MIN_SEA_SPEED_KN = 2.0
MIN_SEA_DAYS = 0.05
GHOST_BUNKER_TOLERANCE_MT = -2.0
PORT_GHOST_BUNKER_TOLERANCE_MT = -5.0

REQUIRED_RAW_COLS = [
    'FO_A', 'FO_L', 'MGO_A', 'MGO_L', 
    'Bunk_FO', 'Bunk_MGO', 'Bunk_MELO', 'Bunk_HSCYLO', 'Bunk_LSCYLO', 'Bunk_GELO', 'Bunk_CYLO', 
    'MELO_R', 'HSCYLO_R', 'LSCYLO_R', 'GELO_R', 'CYLO_R', 
    'Speed', 'DistLeg', 'TotalDist', 'CargoQty', 'Voy', 'Port', 'AD', 'Date', 'Time'
]

def _u(s): return f"data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}"

LOGO_SVG = base64.b64encode(b'<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#c9a84c"/><stop offset="50%" stop-color="#00e0b0"/><stop offset="100%" stop-color="#005f73"/></linearGradient></defs><circle cx="24" cy="24" r="22" fill="none" stroke="url(#pg)" stroke-width="0.8" opacity=".3"/><path d="M24 6L24 42" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"/><path d="M12 24Q24 32 36 24" fill="none" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"/></svg>').decode()

ICONS = {
    "VERIFIED": _u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><circle cx="14" cy="14" r="12" fill="none" stroke="#00e0b0" stroke-width="1" opacity=".2"/><circle cx="14" cy="14" r="7.5" fill="#061a14" stroke="#00e0b0" stroke-width="1.2"/><polyline points="10,14.5 12.8,17 18,10.5" fill="none" stroke="#00e0b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "GHOST BUNKER": _u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><circle cx="14" cy="14" r="12" fill="none" stroke="#e63946" stroke-width="1" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="7.5" fill="#1a0508" stroke="#e63946" stroke-width="1.2"/><g stroke="#e63946" stroke-width="2" stroke-linecap="round"><line x1="11" y1="11" x2="17" y2="17"/><line x1="17" y1="11" x2="11" y2="17"/></g></svg>'),
    "STAT OUTLIER": _u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="20" height="20" rx="5" fill="none" stroke="#7b68ee" stroke-width="1.2"/><circle cx="14" cy="14" r="4.5" fill="#0e0a1e" stroke="#7b68ee" stroke-width="1.2"/><circle cx="14" cy="14" r="1.8" fill="#7b68ee"/></svg>')
}

STATUS_COLORS = {"VERIFIED": "#00e0b0", "GHOST BUNKER": "#e63946", "STAT OUTLIER": "#7b68ee"}

CSS = '''<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500;600&family=Hanken+Grotesk:wght@300;400;500;600;700&display=swap');
:root{--bg:#020609;--s1:#080d14;--s2:#0c1219;--b1:rgba(201,168,76,0.06);--b2:rgba(201,168,76,0.15);--acc:#00e0b0;--acc2:#c9a84c;--red:#e63946;--amber:#d4a843;--purple:#7b68ee;--t1:#dce8f0;--t2:#6d8599;--t3:#3a4d5e;--r:12px;--fd:'Bricolage Grotesque',sans-serif;--fb:'Hanken Grotesk',sans-serif;--fm:'Geist Mono',monospace}
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
div[data-testid="stMetric"]{background:linear-gradient(180deg,var(--s1),var(--s2))!important;border:1px solid var(--b1)!important;border-radius:var(--r);padding:18px 15px!important;position:relative;}
div[data-testid="stMetricLabel"]{font-size:.6rem!important;color:var(--t2)!important;text-transform:uppercase!important;letter-spacing:.12em!important;font-weight:600!important;font-family:var(--fm)!important; white-space: normal; word-wrap: break-word;}
div[data-testid="stMetricValue"]{font-size:1.4rem!important;font-weight:800!important;color:#fff!important;line-height:1.1!important;margin-top:6px!important;font-family:var(--fd)!important; white-space: normal; word-wrap: break-word;}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;border-bottom:2px solid transparent;border-radius:0;padding:12px 18px;color:var(--t3);font-weight:600;font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;font-family:var(--fm);transition:all .3s}
.stTabs [aria-selected="true"]{color:var(--acc)!important;border-bottom-color:var(--acc)!important}
.stDataFrame{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--b1)!important}
.acard{background:var(--s1);border-radius:10px;padding:16px 20px;margin-bottom:8px;border-left:3px solid var(--red); transition:transform .2s}
.pill{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.55rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-family:var(--fm)}
.p-ok{background:rgba(0,224,176,.06);color:var(--acc);border:1px solid rgba(0,224,176,.15)}
.p-c{background:rgba(230,57,70,.06);color:var(--red);border:1px solid rgba(230,57,70,.15)}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--t3);border-radius:2px}
</style>'''
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: PARSER & SCHEMA ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def _sn(val):
    if pd.isna(val): return np.nan
    s = re.sub(r'[^\d.\-]', '', str(val).strip())
    try: return float(s) if s and s not in ('.','-','-.') else np.nan
    except ValueError: return np.nan

def _sn0(val):
    v = _sn(val); return 0.0 if np.isnan(v) else v

def _parse_dt(d_val, t_val):
    try:
        if pd.isna(d_val): return pd.NaT
        ds = str(d_val).strip()
        ds = re.sub(r'20224','2024',ds); ds = re.sub(r'20023','2023',ds)
        ds = re.sub(r'(\d+)\s+([A-Za-z]+)\.?\s+(\d{4})', lambda m:f"{m.group(3)}-{m.group(2)[:3]}-{m.group(1).zfill(2)}", ds)
        p = pd.to_datetime(ds, errors='coerce')
        if pd.isna(p): return pd.NaT
        d_str = p.strftime('%Y-%m-%d')
        
        t_str = '00:00'
        if not pd.isna(t_val):
            tr = re.sub(r'[HhLlTtUuCc\s]', '', str(t_val).strip())
            m = re.match(r'^(\d{1,2}):(\d{2})', tr)
            if m: t_str = f"{m.group(1).zfill(2)}:{m.group(2)}"
        return pd.to_datetime(f"{d_str} {t_str}", errors='coerce')
    except Exception: return pd.NaT

def enforce_schema(df):
    missing = [col for col in REQUIRED_RAW_COLS if col not in df.columns]
    for req in missing:
        if req in ['FO_A', 'FO_L', 'MGO_A', 'MGO_L', 'MELO_R', 'HSCYLO_R', 'LSCYLO_R', 'GELO_R', 'CYLO_R']:
            df[req] = np.nan
        elif req in ['Voy', 'Port', 'AD', 'Date', 'Time']:
            df[req] = ''
        else:
            df[req] = 0.0
    return df

def semantic_parse(uploaded_file):
    vn_raw = re.sub(r'\.[^.]+$', '', uploaded_file.name).strip()
    vname = re.sub(r'[_\-]+', ' ', vn_raw).upper()
    
    if uploaded_file.name.lower().endswith('.xlsx'): df_raw = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
    else: df_raw = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('latin-1', errors='replace')), header=None, on_bad_lines='skip')
        
    if df_raw.empty or len(df_raw) < 4: raise ValueError("File is empty or severely malformed.")

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
                c_comb = f"{c1} {c2}".strip()
                
                if 'VOY' in c_comb: cols_found['Voy'] = j
                elif 'PORT' in c_comb or 'LOC' in c_comb: cols_found['Port'] = j
                elif 'A/D' in c_comb or c_comb == 'AD' or 'STATUS' in c_comb: cols_found['AD'] = j
                elif 'SPEED' in c_comb: cols_found['Speed'] = j
                elif 'CARGO' in c_comb or 'QTY' in c_comb: cols_found['CargoQty'] = j
                elif 'DATE' in c_comb or 'DAY' in c_comb: cols_found['Date'] = j
                elif 'TIME' in c_comb and 'TOTAL' not in c_comb: cols_found['Time'] = j
                elif 'DIST' in c_comb and 'LEG' in c_comb: cols_found['DistLeg'] = j
                elif 'DIST' in c_comb and 'TOTAL' in c_comb: cols_found['TotalDist'] = j
                elif 'BUNKER' in c1 or 'RECEIV' in c1:
                    if 'FO' in c2 and 'MGO' not in c2: cols_found['Bunk_FO'] = j
                    elif 'MGO' in c2: cols_found['Bunk_MGO'] = j
                    elif 'MELO' in c2: cols_found['Bunk_MELO'] = j
                    elif 'HSCYLO' in c2 or 'HS CYL' in c2: cols_found['Bunk_HSCYLO'] = j
                    elif 'LSCYLO' in c2 or 'LS CYL' in c2: cols_found['Bunk_LSCYLO'] = j
                    elif 'CYLO' in c2 or 'CYL OIL' in c2: cols_found['Bunk_CYLO'] = j
                    elif 'GELO' in c2: cols_found['Bunk_GELO'] = j
                elif 'ROB' in c1 or 'REMAIN' in c1:
                    if 'FO A' in c2 or 'FO ACT' in c2: cols_found['FO_A'] = j
                    elif 'FO L' in c2 or 'FO LED' in c2: cols_found['FO_L'] = j
                    elif 'MGO A' in c2: cols_found['MGO_A'] = j
                    elif 'MGO L' in c2: cols_found['MGO_L'] = j
                    elif 'MELO' in c2: cols_found['MELO_R'] = j
                    elif 'HSCYLO' in c2 or 'HS CYL' in c2: cols_found['HSCYLO_R'] = j
                    elif 'LSCYLO' in c2 or 'LS CYL' in c2: cols_found['LSCYLO_R'] = j
                    elif 'CYLO' in c2 or 'CYL OIL' in c2: cols_found['CYLO_R'] = j
                    elif 'GELO' in c2: cols_found['GELO_R'] = j
            break
            
    df = df_raw.iloc[header_idx+1:].copy().reset_index(drop=True)
    for std_name, exc_idx in cols_found.items():
        if std_name in ['Voy', 'Port', 'AD', 'Date', 'Time']: df[std_name] = df.iloc[:, exc_idx]
        else: df[std_name] = df.iloc[:, exc_idx]
            
    df = enforce_schema(df)
    df['Datetime'] = df.apply(lambda r: _parse_dt(r.get('Date'), r.get('Time')), axis=1)
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    df['AD'] = df['AD'].apply(lambda v: 'D' if str(v).upper().strip() in ['D','DEP','SBE','FAOP'] else ('A' if str(v).upper().strip().startswith('A') else v))
    return df, vname

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: FORENSICS & STATE MACHINE (With Sequence Smoothing)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dqi(r1, r2, days, phys_burn, drift):
    if days <= 0 or pd.isna(phys_burn): return 0
    scores = []
    scores.append(100.0) 
    if phys_burn >= GHOST_BUNKER_TOLERANCE_MT: scores.append(100.0)
    else: scores.append(max(0.0, 100 - abs(phys_burn)*5))
    tol = max(30.0, 0.03 * max(_sn0(r1.get('FO_A')), _sn0(r2.get('FO_A'))))
    if tol > 0: scores.append(math.exp(-0.5 * ((drift) / tol)**2) * 100)
    else: scores.append(0.0)
    return int(sum(scores) / len(scores))

def build_state_machine(df):
    ad_events = df[df['AD'].isin(['A', 'D'])].copy()
    if len(ad_events) < 2: raise ValueError("Insufficient A/D events to construct a timeline.")
    
    # THE EVENT AGGREGATOR (Sequence Smoother)
    # This prevents the QUARANTINE_SEQ bug by merging consecutive identical states.
    ad_events['Prev_AD'] = ad_events['AD'].shift(1)
    ad_events = ad_events[ad_events['AD'] != ad_events['Prev_AD']].drop(columns=['Prev_AD']).copy()
    
    trips, cum_drift = [], []
    
    for i in range(len(ad_events)-1):
        r1, r2 = ad_events.iloc[i], ad_events.iloc[i+1]
        idx1, idx2 = r1.name, r2.name
        status, flags = 'VERIFIED', []
        phys_burn, log_burn, drift, days = np.nan, np.nan, np.nan, 0.0
        
        # State Assignment (Guaranteed by the Aggregator above)
        phase = 'SEA' if r1['AD'] == 'D' else 'PORT'
            
        # THE TIME FALLBACK (Fixes the QUARANTINE_TIME bug)
        days = (r2['Datetime'] - r1['Datetime']).total_seconds() / 86400.0
        if days <= 0: 
            days = 0.02  # ~30 minute fallback to prevent zero-division crashes
            flags.append("Time Delta Fallback Applied")
                
        # Physical ROB Validation
        start_rob, end_rob = _sn(r1.get('FO_A')), _sn(r2.get('FO_A'))
        if pd.isna(start_rob) or pd.isna(end_rob):
            status = 'QUARANTINE_ROB'; flags.append("Missing Physical Tank Sounding")
            
        if r1['AD'] == 'D' and not pd.isna(start_rob):
            fol = _sn(r1.get('FO_L'))
            cum_drift.append({'dt': r1['Datetime'], 'gap': start_rob - (fol if not pd.isna(fol) else start_rob), 'port': str(r1.get('Port',''))[:20]})
            
        window = df.loc[idx1+1:idx2]
        
        if phase == 'PORT':
            bfo = _sn0(df.loc[idx1:idx2, 'Bunk_FO'].sum())
            b_melo, b_hscylo, b_lscylo, b_cylo, b_gelo = _sn0(df.loc[idx1:idx2, 'Bunk_MELO'].sum()), _sn0(df.loc[idx1:idx2, 'Bunk_HSCYLO'].sum()), _sn0(df.loc[idx1:idx2, 'Bunk_LSCYLO'].sum()), _sn0(df.loc[idx1:idx2, 'Bunk_CYLO'].sum()), _sn0(df.loc[idx1:idx2, 'Bunk_GELO'].sum())
        else:
            bfo = _sn0(window['Bunk_FO'].sum())
            b_melo, b_hscylo, b_lscylo, b_cylo, b_gelo = _sn0(window['Bunk_MELO'].sum()), _sn0(window['Bunk_HSCYLO'].sum()), _sn0(window['Bunk_LSCYLO'].sum()), _sn0(window['Bunk_CYLO'].sum()), _sn0(window['Bunk_GELO'].sum())
            
        dist = _sn0(window['DistLeg'].sum())
        if dist <= 0 and phase == 'SEA': dist = max(0, _sn0(r2.get('TotalDist')) - _sn0(r1.get('TotalDist')))
        
        speed = window['Speed'].replace(0, np.nan).mean() if not window['Speed'].empty else np.nan
        if pd.isna(speed): speed = dist / (days * 24.0) if days > 0 else 0.0
        
        melo_c = max(0, (_sn0(r1.get('MELO_R')) - _sn0(r2.get('MELO_R'))) + b_melo)
        hscylo_c = max(0, (_sn0(r1.get('HSCYLO_R')) - _sn0(r2.get('HSCYLO_R'))) + b_hscylo)
        lscylo_c = max(0, (_sn0(r1.get('LSCYLO_R')) - _sn0(r2.get('LSCYLO_R'))) + b_lscylo)
        cylo_gen_c = max(0, (_sn0(r1.get('CYLO_R')) - _sn0(r2.get('CYLO_R'))) + b_cylo)
        gelo_c = max(0, (_sn0(r1.get('GELO_R')) - _sn0(r2.get('GELO_R'))) + b_gelo)

        dqi = 0
        if status == 'VERIFIED':
            phys_burn = (start_rob - end_rob) + bfo
            log_start = _sn(r1.get('FO_L')) if not pd.isna(_sn(r1.get('FO_L'))) else start_rob
            log_end = _sn(r2.get('FO_L')) if not pd.isna(_sn(r2.get('FO_L'))) else end_rob
            log_burn = (log_start - log_end) + bfo
            drift = phys_burn - log_burn
            
            if phase == 'PORT' and phys_burn < PORT_GHOST_BUNKER_TOLERANCE_MT:
                status = 'GHOST BUNKER'; flags.append("Mass Imbalance: Missing Port Bunker Receipt")
            elif phase == 'SEA' and phys_burn < GHOST_BUNKER_TOLERANCE_MT:
                status = 'GHOST BUNKER'; flags.append("Negative Sea Burn Impossibility")
                
            dqi = compute_dqi(r1, r2, days, phys_burn, drift)
                
        trips.append({
            'Indicator': ICONS.get(status, ICONS['VERIFIED']) if 'QUARANTINE' not in status else '⛔',
            'Timeline': f"{r1['Datetime'].strftime('%d %b %y')} → {r2['Datetime'].strftime('%d %b %y')}",
            'Date_Start_TS': r1['Datetime'],
            'Phase': phase,
            'Condition': 'LADEN' if _sn0(r1.get('CargoQty', 0)) > 100 else 'BALLAST',
            'Voy': str(r1.get('Voy','')).strip(),
            'Route': f"{str(r1.get('Port',''))[:15]} → {str(r2.get('Port',''))[:15]}" if phase == 'SEA' else f"Port Idle: {str(r1.get('Port',''))[:15]}",
            'Days': round(days, 2),
            'Dist_NM': round(dist, 0),
            'Speed_kn': round(speed, 1),
            'CargoQty': _sn0(r1.get('CargoQty', 0)),
            'FO_A_Start': start_rob if status == 'VERIFIED' else np.nan,
            'Bunk_FO': bfo,
            'FO_A_End': end_rob if status == 'VERIFIED' else np.nan,
            'Phys_Burn': round(phys_burn, 1),
            'Log_Burn': round(log_burn, 1),
            'Drift_MT': round(drift, 1),
            'Daily_Burn': round(phys_burn/days, 1) if days > 0 and not pd.isna(phys_burn) else np.nan,
            'MELO_L': round(melo_c, 0),
            'HSCYLO_L': round(hscylo_c, 0),
            'LSCYLO_L': round(lscylo_c, 0),
            'CYLO_GEN_L': round(cylo_gen_c, 0),
            'GELO_L': round(gelo_c, 0),
            'DQI': int(dqi),
            'Status': status,
            'Flags': ', '.join(flags) if flags else ''
        })

    trip_df = pd.DataFrame(trips)
    
    if len(trip_df) >= 4:
        for cond in ['LADEN', 'BALLAST']:
            ver = trip_df[(trip_df['Status'] == 'VERIFIED') & (trip_df['Phase'] == 'SEA') & (trip_df['Phys_Burn'] > 0) & (trip_df['Condition'] == cond)]
            if len(ver) >= 4:
                q1, q3 = ver['Daily_Burn'].quantile(0.25), ver['Daily_Burn'].quantile(0.75); iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 2.0*iqr, q3 + 2.0*iqr
                    mask = (trip_df['Status'] == 'VERIFIED') & (trip_df['Phase'] == 'SEA') & (trip_df['Condition'] == cond) & ((trip_df['Daily_Burn'] < lo) | (trip_df['Daily_Burn'] > hi))
                    trip_df.loc[mask, 'Status'] = 'STAT OUTLIER'; trip_df.loc[mask, 'Indicator'] = ICONS['STAT OUTLIER']
                    
    return trip_df, cum_drift

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: INTELLIGENCE & AI (Conformal Physics)
# ═══════════════════════════════════════════════════════════════════════════════
def execute_ai_physics(trip_df):
    ai_status_msg = "AI Optimized."
    if not HAS_ML: return trip_df, "AI Offline: xgboost/shap libraries not installed."
    if trip_df.empty: return trip_df, "AI Offline: Empty ledger."
    
    for col in ['AI_Exp', 'Stoch_Var', 'SHAP_Base', 'SHAP_Propulsion', 'SHAP_Mass', 'SHAP_Kinematics', 'SHAP_Season', 'Exp_Lower', 'Exp_Upper']:
        if col not in trip_df.columns: trip_df[col] = np.nan
        
    try:
        sea_mask = (trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Speed_kn'] > MIN_SEA_SPEED_KN)
        if sea_mask.sum() < 4: raise ValueError(f"Insufficient valid Sea Legs ({sea_mask.sum()}). Minimum 4 required to train Conformal Bounds.")
            
        ml = trip_df.loc[sea_mask].copy()
        ml['True_Mass'] = ml['CargoQty'].fillna(0) + ml['FO_A_Start'].fillna(0)
        ml['SOG'] = ml['Dist_NM'] / np.maximum(ml['Days']*24, 0.1)
        ml['Kin_Delta'] = (ml['Speed_kn'] - ml['SOG']).clip(-3.0, 3.0)
        ml['Accel_Penalty'] = ml['Speed_kn'].diff().fillna(0.0).clip(-2.0, 2.0)
        ml['Speed_Cubed'] = ml['Speed_kn']**3
        ml['Season_Sin'] = np.sin(2*np.pi*ml['Date_Start_TS'].dt.month.fillna(6)/12.0)
        
        features = ['Speed_kn', 'Speed_Cubed', 'True_Mass', 'Kin_Delta', 'Accel_Penalty', 'Season_Sin']
        ml[features] = ml[features].fillna(0.0)
        
        X_train = ml[features]
        y_train = ml['Daily_Burn']
        weights = ml['Days'].clip(0.1, 30.0)
        
        if y_train.var() < 0.1: raise ValueError("Target variance too low. Vessel burning identical fuel daily.")

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.06, random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        train_preds = model.predict(X_train)
        residuals = np.abs(y_train - train_preds)
        
        var_model = XGBRegressor(n_estimators=40, max_depth=2, learning_rate=0.05, random_state=42)
        var_model.fit(X_train, residuals, sample_weight=weights)
        
        preds = model.predict(X_train)
        stoch_margin = np.maximum(var_model.predict(X_train) * 1.645, 0.5) 
        
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_train)
        base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        
        trip_df.loc[sea_mask, 'AI_Exp'] = preds.round(1)
        trip_df.loc[sea_mask, 'Stoch_Var'] = stoch_margin.round(1)
        trip_df.loc[sea_mask, 'SHAP_Base'] = base_val
        trip_df.loc[sea_mask, 'SHAP_Propulsion'] = sv[:,0] + sv[:,1] 
        trip_df.loc[sea_mask, 'SHAP_Mass'] = sv[:,2]
        trip_df.loc[sea_mask, 'SHAP_Kinematics'] = sv[:,3] + sv[:,4] 
        trip_df.loc[sea_mask, 'SHAP_Season'] = sv[:,5]
        trip_df.loc[sea_mask, 'Exp_Lower'] = preds - stoch_margin
        trip_df.loc[sea_mask, 'Exp_Upper'] = preds + stoch_margin
        
        outlier_mask = sea_mask & ((trip_df['Daily_Burn'] < trip_df['Exp_Lower']) | (trip_df['Daily_Burn'] > trip_df['Exp_Upper']))
        trip_df.loc[outlier_mask, 'Status'] = 'STAT OUTLIER'
        
    except ValueError as e: ai_status_msg = f"AI Offline: {str(e)}"
    except Exception as e:
        ai_status_msg = f"AI Critical Exception: {str(e)}"
        print(traceback.format_exc())
    return trip_df, ai_status_msg

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: APP & UI ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline(uploaded_file):
    try:
        parsed_df, vname = semantic_parse(uploaded_file)
        trip_df, cum_drift = build_state_machine(parsed_df)
        trip_df, ai_msg = execute_ai_physics(trip_df)
        
        quarantined = len(trip_df[trip_df['Status'].str.contains('QUARANTINE')])
        valid_sea = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED')]
        avg_sea = valid_sea['Phys_Burn'].sum() / valid_sea['Days'].sum() if valid_sea['Days'].sum() > 0 else 0.0
        trip_df['Total_CYLO'] = trip_df.get('HSCYLO_L',0) + trip_df.get('LSCYLO_L',0) + trip_df.get('CYLO_GEN_L',0)
        
        summary = {
            'vname': vname,
            'integrity': round((len(trip_df) - quarantined) / len(trip_df) * 100, 1) if not trip_df.empty else 0,
            'avg_dqi': round(trip_df['DQI'].mean(), 0) if not trip_df.empty else 0,
            'total_fuel': round(trip_df['Phys_Burn'].sum(skipna=True), 1),
            'avg_sea_burn': round(avg_sea, 1),
            'total_nm': round(trip_df['Dist_NM'].sum(), 0),
            'total_days': round(trip_df['Days'].sum(), 1),
            'total_melo': round(trip_df.get('MELO_L', pd.Series([0])).sum(), 0),
            'total_cylo': round(trip_df['Total_CYLO'].sum(), 0),
            'cycles': len(trip_df),
            'quarantined': quarantined,
            'anomalies': len(trip_df[trip_df['Status'].isin(['GHOST BUNKER', 'STAT OUTLIER'])]),
            'ai_msg': ai_msg
        }
        return trip_df, summary, cum_drift, None
    except ValueError as e: return pd.DataFrame(), None, None, f"Parsing Rejected: {str(e)}"
    except Exception as e: return pd.DataFrame(), None, None, f"System Crash: {str(e)}"

_BL=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',hovermode='x unified',font=dict(family='Hanken Grotesk',color='#dce8f0'),margin=dict(l=0,r=0,t=55,b=20))
_AX=dict(gridcolor='rgba(201,168,76,0.04)',zerolinecolor='rgba(201,168,76,0.06)',tickfont=dict(size=10))

def chart_fuel(df):
    sea = df[(df['Phase'] == 'SEA') & (~df['Status'].str.contains('QUARANTINE'))]
    port = df[(df['Phase'] == 'PORT') & (~df['Status'].str.contains('QUARANTINE'))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08)
    if not sea.empty: fig.add_trace(go.Bar(x=sea['Timeline'], y=sea['Phys_Burn'], name='Sea Fuel', marker_color='rgba(0,224,176,0.15)', marker_line_color='#00e0b0', marker_line_width=1.5), row=1, col=1)
    if not port.empty: fig.add_trace(go.Bar(x=port['Timeline'], y=port['Phys_Burn'], name='Port Fuel', marker_color='rgba(123,104,238,0.15)', marker_line_color='#7b68ee', marker_line_width=1.5), row=1, col=1)
    if not sea.empty: fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Daily_Burn'], name='Sea MT/day', mode='lines+markers', line=dict(color='#00e0b0', width=2, shape='spline')), row=1, col=1)
    if not sea.empty: fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Speed_kn'], name='Sea Speed', mode='lines+markers', line=dict(color='#c9a84c', width=2, shape='spline')), row=2, col=1)
    fig.update_layout(**_BL, title='Tri-State Fuel Consumption & Sea Speed', barmode='group', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(tickangle=-45, automargin=True, **_AX); fig.update_yaxes(**_AX); return fig

def chart_lube(df):
    fig = go.Figure()
    if df.get('MELO_L', pd.Series([0])).sum() > 0: fig.add_trace(go.Bar(x=df['Timeline'], y=df['MELO_L'], name='MELO', marker_color='rgba(0,224,176,0.12)', marker_line_color='#00e0b0', marker_line_width=1.3))
    if df.get('Total_CYLO', pd.Series([0])).sum() > 0: fig.add_trace(go.Bar(x=df['Timeline'], y=df['Total_CYLO'], name='CYLO (All)', marker_color='rgba(123,104,238,0.12)', marker_line_color='#7b68ee', marker_line_width=1.3))
    if df.get('GELO_L', pd.Series([0])).sum() > 0: fig.add_trace(go.Bar(x=df['Timeline'], y=df['GELO_L'], name='GELO', marker_color='rgba(201,168,76,0.12)', marker_line_color='#c9a84c', marker_line_width=1.3))
    fig.update_layout(**_BL, title='Lubricant Consumption (Liters)', barmode='group', showlegend=True, yaxis=dict(title='L', **_AX), xaxis=dict(automargin=True, **_AX))
    fig.update_xaxes(tickangle=-45); return fig

st.markdown(f"""<div class="hero"><div class="hero-left"><img src="data:image/svg+xml;base64,{LOGO_SVG}" class="hero-logo" alt=""/><div><div class="hero-title">POSEIDON TITAN</div><div class="hero-sub">Enterprise Forensic Engine</div></div></div><div class="hero-badge"><span>KERNEL</span>&ensp;Smoothed State Machine<br><span>PIPELINE</span>&ensp;ECA High-Fidelity Schema<br><span>BUILD</span>&ensp;v29.0 Production</div></div>""",unsafe_allow_html=True)

files = st.file_uploader('Upload vessel telemetry', accept_multiple_files=True, type=['xlsx','csv'], label_visibility='collapsed')

if not files:
    st.info("Drop vessel noon-report files to execute the Zero-Tolerance State Machine & Physics isolation.")
    st.stop()

fleet_results = []
for f in files:
    with st.spinner(f'Auditing {f.name}...'):
        try:
            df, sum_data, cum_drift, err = run_pipeline(f)
            if err:
                st.error(f"**Rejected {f.name}:** {err}"); continue
            if df.empty:
                st.warning(f"No valid events extracted from {f.name}. Check template schema."); continue
                
            fleet_results.append({'name': sum_data['vname'], 'summary': sum_data, 'df': df})
            
            ic = STATUS_COLORS['VERIFIED'] if sum_data['integrity'] >= 80 else (STATUS_COLORS['STAT OUTLIER'] if sum_data['integrity'] >= 50 else STATUS_COLORS['GHOST BUNKER'])
            st.markdown(f"""<div class="vcard"><div style="display:flex;justify-content:space-between;align-items:center"><div><div style="font-family:var(--fd);font-weight:800;font-size:1.3rem;color:#fff;letter-spacing:-0.03em">{sum_data['vname']}</div><div style="font-family:var(--fm);font-size:.62rem;color:var(--t2);margin-top:5px;letter-spacing:0.04em">{sum_data['cycles']} LEGS&ensp;·&ensp;{sum_data['total_days']:.0f} DAYS&ensp;·&ensp;{int(sum_data['total_nm']):,} NM</div><div style="font-family:var(--fm);font-size:.55rem;color:var(--amber);margin-top:3px;letter-spacing:0.04em">{sum_data['ai_msg']}</div></div><div style="text-align:right"><div style="font-family:var(--fd);font-weight:800;font-size:1.6rem;color:{ic};margin-top:5px">{sum_data['integrity']:.0f}%</div><div style="font-family:var(--fm);font-size:.5rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em">Audit Integrity</div></div></div></div>""",unsafe_allow_html=True)

            cols = st.columns(6)
            cols[0].metric('Verified Fuel (MT)', f"{sum_data['total_fuel']:,.1f}")
            cols[1].metric('Avg Sea Burn (MT/d)', f"{sum_data['avg_sea_burn']:.1f}")
            cols[2].metric('Total MELO (L)', f"{int(sum_data['total_melo']):,}")
            cols[3].metric('Total CYLO (L)', f"{int(sum_data['total_cylo']):,}")
            cols[4].metric('Mass Anomalies', f"{sum_data['anomalies']}")
            cols[5].metric('Quarantined Legs', f"{sum_data['quarantined']}")

            t1, t2, t3, t4, t5, t6 = st.tabs(['IMMUTABLE LEDGER', 'COMMERCIAL P&L', 'LUBE & DRIFT', 'AI DIGITAL TWIN', 'SHAP EXPLAINER', 'QUARANTINE LOG'])

            with t1:
                st.markdown('<span style="font-family:var(--fm); font-size:0.7rem; color:var(--acc)">[START ROB] + [BUNKERS] - [END ROB] = [PHYSICAL BURN]</span>', unsafe_allow_html=True)
                dcfg={'Indicator':st.column_config.ImageColumn(' '), 'Timeline':st.column_config.TextColumn('TIMELINE',width='medium'), 'Phase':st.column_config.TextColumn('LEG'), 'Days':st.column_config.NumberColumn('DAYS',format='%.2f'), 'Speed_kn':st.column_config.NumberColumn('SPD',format='%.1f'), 'FO_A_Start':st.column_config.NumberColumn('START ROB',format='%.1f'), 'Bunk_FO':st.column_config.NumberColumn('+ BUNKERS',format='%.1f'), 'FO_A_End':st.column_config.NumberColumn('- END ROB',format='%.1f'), 'Phys_Burn':st.column_config.NumberColumn('= BURN',format='%.1f'), 'Log_Burn':st.column_config.NumberColumn('LOG BURN',format='%.1f'), 'DQI':st.column_config.ProgressColumn('DQI',format='%d',min_value=0,max_value=100), 'Daily_Burn':st.column_config.NumberColumn('MT/DAY',format='%.1f'), 'Total_CYLO':st.column_config.NumberColumn('CYLO (ALL)',format='%d'), 'Status':st.column_config.TextColumn('STATUS',width='medium')}
                st.dataframe(df[['Indicator','Timeline','Phase','Days','Speed_kn','FO_A_Start','Bunk_FO','FO_A_End','Phys_Burn','Log_Burn','Drift_MT','Daily_Burn','Total_CYLO','MELO_L','GELO_L','DQI','Status']], column_config=dcfg, hide_index=True, use_container_width=True)
                buf=io.BytesIO(); exp=df.drop(columns=['Indicator','Date_Start_TS'],errors='ignore')
                with pd.ExcelWriter(buf,engine='openpyxl') as w: exp.to_excel(w,index=False,sheet_name='Audit')
                buf.seek(0)
                st.download_button('Export Tri-State Ledger',data=buf,file_name=f"{sum_data['vname'].replace(' ','_')}_LEDGER.xlsx",key=f"dl_{sum_data['vname']}")

            with t2:
                voy = df[~df['Status'].str.contains('QUARANTINE')].groupby('Voy', dropna=False).agg(Total_Fuel=('Phys_Burn','sum'), Sea_Days=('Days', lambda x: x[df.loc[x.index, 'Phase']=='SEA'].sum()), Port_Days=('Days', lambda x: x[df.loc[x.index, 'Phase']=='PORT'].sum()), Sea_Fuel=('Phys_Burn', lambda x: x[df.loc[x.index, 'Phase']=='SEA'].sum()), Bunkers=('Bunk_FO','sum'), Dist=('Dist_NM','sum'), HSCYLO=('HSCYLO_L','sum'), LSCYLO=('LSCYLO_L','sum')).reset_index()
                voy['Sea MT/Day'] = np.where(voy['Sea_Days']>0, voy['Sea_Fuel']/voy['Sea_Days'], 0.0)
                st.dataframe(voy, hide_index=True, use_container_width=True)

            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    if df.get('MELO_L', pd.Series([0])).sum() + df.get('Total_CYLO', pd.Series([0])).sum() > 0: st.plotly_chart(chart_lube(df), use_container_width=True, config={'displayModeBar':False})
                    else: st.info('No lubricant consumption data detected.')
                with c2:
                    if cum_drift:
                        cdf=pd.DataFrame(cum_drift)
                        fig_d=go.Figure()
                        fig_d.add_trace(go.Scatter(x=cdf['dt'],y=cdf['gap'],mode='lines+markers',name='A−L Gap',line=dict(color='#c9a84c',width=2),marker=dict(size=3),fill='tozeroy',fillcolor='rgba(201,168,76,0.04)'))
                        fig_d.add_hline(y=0,line=dict(color='rgba(255,255,255,0.06)',width=1,dash='dot'))
                        fig_d.update_layout(**_BL,title='Cumulative Physical vs Logged Mass Drift',yaxis=dict(title='FO_A − FO_L (MT)',**_AX),xaxis=dict(automargin=True, **_AX)); fig_d.update_xaxes(tickangle=-45); st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar':False})

            with t4:
                st.plotly_chart(chart_fuel(df), use_container_width=True, config={'displayModeBar':False})
                sea_df = df[(df['Phase'] == 'SEA') & (df['Status'] == 'VERIFIED')]
                if 'AI_Exp' in sea_df.columns and sea_df['AI_Exp'].abs().sum() > 0:
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Scatter(x=sea_df['Timeline'].tolist() + sea_df['Timeline'].tolist()[::-1], y=sea_df['Exp_Upper'].tolist() + sea_df['Exp_Lower'].tolist()[::-1], fill='toself', fillcolor='rgba(123,104,238,0.15)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='90% Conformal Interval'))
                    fig_c.add_trace(go.Scatter(x=sea_df['Timeline'], y=sea_df['AI_Exp'], name="Expected Mean", line=dict(color="#7b68ee", width=2, dash='dot')))
                    fig_c.add_trace(go.Scatter(x=sea_df['Timeline'], y=sea_df['Daily_Burn'], name="Audited Burn", mode='lines+markers', line=dict(color="#00e0b0", width=2), marker=dict(size=6, color="#fff")))
                    fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color="#dce8f0", title='Conformal Propulsion Bounds (Verified Sea Legs Only)', height=400, yaxis=dict(title='MT/day', gridcolor='rgba(201,168,76,0.04)'), xaxis=dict(tickangle=-45, automargin=True, gridcolor='rgba(201,168,76,0.04)'))
                    st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar':False})

            with t5:
                sea = df[(df['Phase'] == 'SEA') & (df['Status'] == 'VERIFIED')]
                if 'SHAP_Base' in sea.columns and sea['SHAP_Base'].abs().sum() > 0:
                    options = sea['Timeline'].tolist()
                    sel = st.selectbox('Select Verified Sea Passage', options, key=f'shap_{sum_data["vname"]}')
                    tr = sea[sea['Timeline']==sel].iloc[0]
                    eb = tr['SHAP_Base'] + tr['SHAP_Propulsion'] + tr['SHAP_Mass'] + tr['SHAP_Kinematics'] + tr['SHAP_Season']
                    fig_w = go.Figure(go.Waterfall(name="SHAP", orientation="v", measure=["absolute","relative","relative","relative","relative","total"], x=["Baseline", "Propulsion", "True Mass", "Kinematics", "Season", "Expected"], textposition="outside", text=[f"{tr['SHAP_Base']:.1f}", f"{tr['SHAP_Propulsion']:+.1f}", f"{tr['SHAP_Mass']:+.1f}", f"{tr['SHAP_Kinematics']:+.1f}", f"{tr['SHAP_Season']:+.1f}", f"{eb:.1f}"], y=[tr['SHAP_Base'], tr['SHAP_Propulsion'], tr['SHAP_Mass'], tr['SHAP_Kinematics'], tr['SHAP_Season'], 0], connector={"line":{"color":"rgba(201,168,76,0.15)"}}, decreasing={"marker":{"color":"#00e0b0"}}, increasing={"marker":{"color":"#e63946"}}, totals={"marker":{"color":"#7b68ee"}}))
                    fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, title=dict(text=f"SHAP Audit: {tr['Route']} ({tr['Speed_kn']}kn)", font=dict(color='#fff')), yaxis=dict(gridcolor='rgba(201,168,76,0.04)'))
                    st.plotly_chart(fig_w, use_container_width=True)
                else: st.warning("AI Explainability Offline: See header logs.")

            with t6:
                quar = df[df['Status'].str.contains('QUARANTINE|GHOST')]
                if quar.empty: st.success("Zero anomalies. All timelines and mass balances intact.")
                else:
                    for _,r in quar.iterrows():
                        c = STATUS_COLORS.get(r['Status'], '#e63946')
                        st.markdown(f'<div class="acard"><span style="color:{c};font-weight:700;font-size:.7rem">{r["Status"]}</span><span style="color:var(--t3);margin-left:10px;font-size:.7rem">{r["Timeline"]}</span><div style="color:var(--t2);font-size:.7rem;margin-top:5px">Exception: {r["Flags"]}</div></div>', unsafe_allow_html=True)
            st.divider()
            
        except Exception as e:
            st.error(f'System Failure on file: {f.name} - Exception: {str(e)}')
            with st.expander('View Traceback'): st.code(traceback.format_exc())

if len(fleet_results) > 1:
    st.markdown('<h2 style="color:#fff;font-family:var(--fd);margin-top:10px">Fleet Comparison Matrix</h2>',unsafe_allow_html=True)
    fleet_rows=[{'Vessel':r['name'],'Legs':r['summary']['cycles'],'Verified':f"{r['summary']['integrity']:.1f}%",'DQI':int(r['summary']['avg_dqi']),'Fuel MT':r['summary']['total_fuel'],'Sea Burn':r['summary']['avg_sea_burn'],'Anomalies':r['summary']['anomalies'],'NM':int(r['summary']['total_nm'])} for r in fleet_results]
    st.dataframe(pd.DataFrame(fleet_rows), hide_index=True, use_container_width=True)
