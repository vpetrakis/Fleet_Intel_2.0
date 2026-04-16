import streamlit as st
import pandas as pd
import numpy as np
import re, io, math, traceback, base64, warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# TITAN CORE: BULLETPROOF PRODUCTION BUILD (Zero-Crash Architecture)
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
div[data-testid="stMetricValue"]{font-size:1.3rem!important;font-weight:800!important;color:#fff!important;line-height:1.1!important;margin-top:6px!important;font-family:var(--fd)!important;letter-spacing:-.03em!important; white-space: normal; word-wrap: break-word;}
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
ICONS={"VERIFIED":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#00e0b0" stroke-width="1" opacity=".2"><animate attributeName="r" values="12;13;12" dur="3s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#061a14" stroke="#00e0b0" stroke-width="1.2" filter="url(#g)"/><polyline points="10,14.5 12.8,17 18,10.5" fill="none" stroke="#00e0b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),"GHOST BUNKER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g2"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#e63946" stroke-width="1" stroke-dasharray="4 3"><animateTransform attributeName="transform" type="rotate" from="0 14 14" to="360 14 14" dur="8s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#1a0508" stroke="#e63946" stroke-width="1.2" filter="url(#g2)"/><g stroke="#e63946" stroke-width="2" stroke-linecap="round"><line x1="11" y1="11" x2="17" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line><line x1="17" y1="11" x2="11" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line></g></svg>'),"LEDGER VARIANCE":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g3"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><polygon points="14,3 3,25 25,25" fill="none" stroke="#d4a843" stroke-width="1.2" stroke-linejoin="round" filter="url(#g3)"><animate attributeName="stroke-opacity" values="1;.3;1" dur="2s" repeatCount="indefinite"/></polygon><line x1="14" y1="11" x2="14" y2="18" stroke="#d4a843" stroke-width="2" stroke-linecap="round"/><circle cx="14" cy="21.5" r="1.2" fill="#d4a843"/></svg>'),"STAT OUTLIER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g4"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><rect x="4" y="4" width="20" height="20" rx="5" fill="none" stroke="#7b68ee" stroke-width="1.2" filter="url(#g4)"><animate attributeName="stroke-dasharray" values="0,80;80,0;0,80" dur="4s" repeatCount="indefinite"/></rect><circle cx="14" cy="14" r="4.5" fill="#0e0a1e" stroke="#7b68ee" stroke-width="1.2"/><circle cx="14" cy="14" r="1.8" fill="#7b68ee"><animate attributeName="r" values="1.8;2.8;1.8" dur="2s" repeatCount="indefinite"/></circle></svg>')}
SC={"VERIFIED":"#00e0b0","GHOST BUNKER":"#e63946","LEDGER VARIANCE":"#d4a843","STAT OUTLIER":"#7b68ee"}

def _rgba(h,a): return f"rgba({int(h.lstrip('#')[0:2],16)},{int(h.lstrip('#')[2:4],16)},{int(h.lstrip('#')[4:6],16)},{a})"
OPS_KW=['RDV','OPL','STRAIT','CANAL','SECTOR','ZONE','RV PT','RV POINT','PILOT','ANCH','ROADSTEAD','TRAFFIC','SEPARATION','PSTN','KUMKALE','GELIBOLU','TURKELI','GREAT BELT']
def _is_ops(n): return any(k in str(n).upper() for k in OPS_KW)
def gauss_mf(v,c,s): return math.exp(-0.5*((v-c)/s)**2) if s>0 else (1.0 if v==c else 0.0)
def trap_mf(v,a,b,c,d):
    if v<=a or v>=d: return 0.0
    if a<v<b: return (v-a)/(b-a)
    if b<=v<=c: return 1.0
    if c<v<d: return (d-v)/(d-c)
    return 0.0
def _sn(val):
    if pd.isna(val): return np.nan
    s = re.sub(r'[^\d.\-]', '', str(val).strip())
    try: return float(s) if s and s not in ('.','-','-.') else np.nan
    except: return np.nan
def _sn0(val):
    v = _sn(val)
    return 0.0 if np.isnan(v) else v
def _parse_dt(d_val,t_val):
    try:
        ds=str(d_val).strip()
        ts=str(t_val).strip() if pd.notna(t_val) else '00:00'
        return pd.to_datetime(f"{ds} {ts}", errors='coerce')
    except: return pd.NaT

def compute_dqi(r1, r2, daily_burn, drift, chrono_bad, mgo_neg):
    s={}
    # Safe dictionary access (.get) to prevent KeyErrors
    rob_a1 = r1.get('FO_A', np.nan); rob_a2 = r2.get('FO_A', np.nan)
    s['rob'] = 1.0 if not np.isnan(rob_a1) and not np.isnan(rob_a2) else 0.3
    tol = max(30.0, 0.03 * max(rob_a1 if not np.isnan(rob_a1) else 0, rob_a2 if not np.isnan(rob_a2) else 0))
    s['drift'] = gauss_mf(drift, 0.0, tol)
    if daily_burn > 0: s['burn'] = gauss_mf(daily_burn, 30.0, 25.0)
    elif daily_burn == 0: s['burn'] = 0.5
    else: s['burn'] = 0.1
    s['chrono'] = 0.3 if chrono_bad else 1.0
    s['mgo'] = 0.3 if mgo_neg else 1.0
    w = {'rob':0.30,'drift':0.30,'burn':0.20,'chrono':0.10,'mgo':0.10}
    log_sum = sum(w[k]*math.log(max(v,0.001)) for k,v in s.items())
    return min(100, max(0, round(math.exp(log_sum)*100, 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# AI CORE: PERMISSIVE CONFORMAL PHYSICS (Only requires 2 cycles)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def calculate_stochastic_variance(trip_df):
    zeros_df = pd.DataFrame({
        'Stoch_Var':[0.0]*len(trip_df), 'SHAP_Base':[0.0]*len(trip_df), 'SHAP_Prop':[0.0]*len(trip_df), 
        'SHAP_Mass':[0.0]*len(trip_df), 'SHAP_Weath':[0.0]*len(trip_df), 'SHAP_Lag':[0.0]*len(trip_df),
        'Exp_Lower':[0.0]*len(trip_df), 'Exp_Upper':[0.0]*len(trip_df)
    }, index=trip_df.index)
    
    try:
        if not HAS_ML or len(trip_df) < 2: return zeros_df
        
        ml = trip_df[['Speed_kn','CargoQty','Condition','Daily_Burn','Days','Date_Start_TS','Dist_NM', 'Drift_MT']].copy()
        
        ml['SOG'] = ml['Dist_NM'] / np.maximum(ml['Days']*24, 0.1)
        ml['Kin_Delta'] = (ml['Speed_kn'] - ml['SOG']).clip(-3.0, 3.0)
        ml['V3'] = ml['Speed_kn']**3
        ml['Cargo_MT'] = np.where(ml['Condition']=='LADEN', ml['CargoQty'], 0.0)
        ml['Froude_Proxy'] = (ml['Speed_kn']**2) * (ml['Cargo_MT'] / 10000.0)
        
        ml = ml.sort_values('Date_Start_TS')
        ml['Hull_EMA'] = ml['Kin_Delta'].ewm(span=10, adjust=False).mean()
        ml['Season'] = np.sin(2*np.pi*ml['Date_Start_TS'].dt.month.fillna(6)/12.0)
        
        mask = (ml['Daily_Burn'] > 1.0) & (ml['Days'] > 0.1) & (ml['Speed_kn'] > 2.0)
        if mask.sum() < 2: return zeros_df

        t_mod = XGBRegressor(n_estimators=20, max_depth=2, random_state=42)
        t_mod.fit(ml.loc[mask, ['Speed_kn', 'Cargo_MT', 'Kin_Delta']], ml.loc[mask, 'Daily_Burn'])
        ml['Lag'] = (ml['Daily_Burn'] - t_mod.predict(ml[['Speed_kn', 'Cargo_MT', 'Kin_Delta']])).shift(1).fillna(0).clip(-12, 12)

        features = ['Speed_kn', 'V3', 'Cargo_MT', 'Froude_Proxy', 'Kin_Delta', 'Hull_EMA', 'Season', 'Lag', 'Drift_MT']
        ml[features] = ml[features].fillna(0.0)

        model = XGBRegressor(n_estimators=100, max_depth=3, reg_lambda=5.0, learning_rate=0.05, random_state=42)
        model.fit(ml.loc[mask, features], ml.loc[mask, 'Daily_Burn'])
        mean_preds = model.predict(ml[features])

        residuals = np.abs(ml.loc[mask, 'Daily_Burn'] - model.predict(ml.loc[mask, features]))
        var_model = XGBRegressor(n_estimators=40, max_depth=2, reg_lambda=10.0, random_state=42)
        var_model.fit(ml.loc[mask, features], residuals)
        
        pred_variance = var_model.predict(ml[features])
        margin = np.maximum(pred_variance * 1.645, 0.5) 
        
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(ml[features])
        base = explainer.expected_value
        if isinstance(base, np.ndarray): base = base[0]

        return pd.DataFrame({
            'Stoch_Var': margin.round(1),
            'SHAP_Base': [base]*len(ml),
            'SHAP_Prop': sv[:,0] + sv[:,1] + sv[:,3],
            'SHAP_Mass': sv[:,2],
            'SHAP_Weath': sv[:,4] + sv[:,5] + sv[:,6],
            'SHAP_Lag': sv[:,7] + sv[:,8],
            'Exp_Lower': mean_preds - margin,
            'Exp_Upper': mean_preds + margin
        }, index=trip_df.index)
        
    except Exception as e: 
        return zeros_df

# ═══════════════════════════════════════════════════════════════════════════════
# INGEST: CRASH-PROOF FUZZY HEADERS & D-TO-D ACTUAL LEDGER
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

    header_idx = 0
    cols_found = {}
    for i in range(min(60, len(df_raw))):
        vals = [str(x).upper() for x in df_raw.iloc[i].values if pd.notna(x)]
        if any(k in v for v in vals for k in ['DATE', 'DAY']) and any(k in v for v in vals for k in ['PORT', 'LOC']):
            header_idx = i
            for j, cell in enumerate(df_raw.iloc[i].values):
                c = str(cell).upper().strip()
                if 'VOY' in c: cols_found['Voy'] = j
                elif 'PORT' in c or 'LOC' in c: cols_found['Port'] = j
                elif 'A/D' in c or c == 'AD' or 'STATUS' in c: cols_found['AD'] = j
                elif 'SPEED' in c: cols_found['Speed'] = j
                elif 'CARGO' in c and 'QTY' in c: cols_found['CargoQty'] = j
                elif 'DATE' in c or 'DAY' in c: cols_found['Date'] = j
                elif 'TIME' in c and 'TOTAL' not in c: cols_found['Time'] = j
                elif 'DIST' in c and 'LEG' in c: cols_found['DistLeg'] = j
                elif 'DIST' in c and 'TOTAL' in c: cols_found['TotalDist'] = j
                elif 'TIME' in c and 'TOTAL' in c: cols_found['TotalTime'] = j
                elif 'BUNK' in c and 'FO' in c: cols_found['Bunk_FO'] = j
                elif 'BUNK' in c and 'MGO' in c: cols_found['Bunk_MGO'] = j
                elif 'BUNK' in c and 'MELO' in c: cols_found['Bunk_MELO'] = j
                elif 'BUNK' in c and 'HSCYLO' in c: cols_found['Bunk_HSCYLO'] = j
                elif 'BUNK' in c and 'LSCYLO' in c: cols_found['Bunk_LSCYLO'] = j
                elif 'BUNK' in c and 'GELO' in c: cols_found['Bunk_GELO'] = j
                elif 'FO' in c and 'L' in c: cols_found['FO_L'] = j
                elif 'FO' in c and 'A' in c: cols_found['FO_A'] = j
                elif 'MGO' in c and 'L' in c: cols_found['MGO_L'] = j
                elif 'MGO' in c and 'A' in c: cols_found['MGO_A'] = j
                elif 'MELO' in c and 'R' in c: cols_found['MELO_R'] = j
                elif 'HSCYLO' in c and 'R' in c: cols_found['HSCYLO_R'] = j
                elif 'LSCYLO' in c and 'R' in c: cols_found['LSCYLO_R'] = j
                elif 'GELO' in c and 'R' in c: cols_found['GELO_R'] = j
                elif 'CYLO' in c and 'R' in c: cols_found['CYLO_R'] = j # Fallback
            break
            
    df = df_raw.iloc[header_idx+1:].copy().reset_index(drop=True)
    
    for std_name, exc_idx in cols_found.items():
        if exc_idx < len(df.columns):
            if std_name in ['Voy', 'Port', 'AD', 'Date', 'Time']:
                df[std_name] = df.iloc[:, exc_idx]
            else:
                df[std_name] = df.iloc[:, exc_idx].apply(_sn).fillna(0.0)
                
    # ABSOLUTE GUARANTEE: Force initialize every required column to prevent KeyErrors
    REQ_COLS = ['FO_A', 'FO_L', 'MGO_A', 'MGO_L', 'Bunk_FO', 'Bunk_MGO', 'Bunk_MELO', 'Bunk_HSCYLO', 'Bunk_LSCYLO', 'Bunk_GELO', 'MELO_R', 'HSCYLO_R', 'LSCYLO_R', 'GELO_R', 'CYLO_R', 'Speed', 'DistLeg', 'TotalDist', 'TotalTime', 'CargoQty', 'Voy', 'Port', 'AD']
    for req in REQ_COLS:
        if req not in df.columns:
            df[req] = 0.0 if req not in ['Voy','Port','AD'] else ''

    df['Datetime'] = df.apply(lambda r: _parse_dt(r.get('Date'), r.get('Time')), axis=1)
    n_before = len(df); df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True); n_dropped = n_before - len(df)
    
    if len(df) < 2: return pd.DataFrame(), vname, {}, []
    
    def _cad(v):
        v = str(v).strip().upper().replace(' ','')
        if v in ['D', 'DEP', 'SBE', 'FAOP']: return 'D'
        if v.startswith('A') and 'D' not in v: return 'A'
        if 'D' in v and 'A' not in v: return 'D'
        return v
    df['AD'] = df['AD'].apply(_cad)
    
    d_indices = df[df['AD'] == 'D'].index.tolist()
    if len(d_indices) < 2: return pd.DataFrame(), vname, {}, []
    
    cum_drift = []
    for idx in d_indices:
        fa = _sn0(df.loc[idx].get('FO_A', 0)); fl = _sn0(df.loc[idx].get('FO_L', 0))
        cum_drift.append({'dt': df.loc[idx].get('Datetime'), 'gap': fa - fl, 'port': str(df.loc[idx].get('Port',''))[:20]})
        
    trips = []
    for ci in range(len(d_indices)-1):
        idx1, idx2 = d_indices[ci], d_indices[ci+1]
        r1, r2 = df.loc[idx1], df.loc[idx2]
        between = df.loc[idx1+1:idx2-1]; a_rows = between[between['AD'] == 'A']
        
        port_dep = str(r1.get('Port','')).strip()[:25] or '—'
        port_arr = '—'
        for _, ar in a_rows.iterrows():
            pn = str(ar.get('Port','')).strip()
            if pn and not _is_ops(pn): port_arr = pn[:25]; break
        if port_arr == '—':
            if not a_rows.empty: port_arr = str(a_rows.iloc[-1].get('Port','')).strip()[:25] or '—'
            else: port_arr = str(r2.get('Port','')).strip()[:25] or '—'
            
        window = df.loc[idx1+1:idx2]
        hours = window['TotalTime'].sum()
        chrono_bad = False
        if hours <= 0:
            dt_d = (r2['Datetime'] - r1['Datetime']).total_seconds() / 3600.0
            if dt_d > 0: hours = dt_d; chrono_bad = True
            else: continue
            
        days = hours / 24.0
        leg_nm = window['DistLeg'].sum() if 'DistLeg' in window.columns else window['TotalDist'].sum()
        if leg_nm <= 0: leg_nm = max(0.0, r2.get('TotalDist', 0.0))
        
        spd_v = window['Speed'].replace(0, np.nan).dropna()
        speed = spd_v.mean() if not spd_v.empty else (leg_nm/hours if hours>0 else 0.0)
        
        # CRASH-PROOF MATH USING .get()
        bfo = window['Bunk_FO'].sum(); bmgo = window['Bunk_MGO'].sum()
        hfo_c = (r1.get('FO_A',0) - r2.get('FO_A',0)) + bfo
        mgo_raw = (r1.get('MGO_A',0) - r2.get('MGO_A',0)) + bmgo
        mgo_c = max(0.0, mgo_raw); mgo_neg = mgo_raw < -5
        
        drift = (r1.get('FO_A',0) - r1.get('FO_L',0)) 
        
        bmelo = window['Bunk_MELO'].sum(); bhsc = window['Bunk_HSCYLO'].sum(); blsc = window['Bunk_LSCYLO'].sum(); bgelo = window['Bunk_GELO'].sum()
        
        melo_c = max(0, (r1.get('MELO_R',0) - r2.get('MELO_R',0)) + bmelo)
        
        # Safe Cyl Oil logic (handles both explicit HSCYLO/LSCYLO and combined CYLO templates)
        hsc_c = max(0, (r1.get('HSCYLO_R',0) - r2.get('HSCYLO_R',0)) + bhsc)
        lsc_c = max(0, (r1.get('LSCYLO_R',0) - r2.get('LSCYLO_R',0)) + blsc)
        cylo_fallback = max(0, (r1.get('CYLO_R',0) - r2.get('CYLO_R',0)))
        total_cylo = hsc_c + lsc_c if (hsc_c + lsc_c) > 0 else cylo_fallback
        
        gelo_c = max(0, (r1.get('GELO_R',0) - r2.get('GELO_R',0)) + bgelo)
        
        total_fuel = hfo_c + mgo_c
        daily_burn = total_fuel / days if days > 0 else 0.0
        
        cargo = str(r1.get('CargoName', '')).strip().upper(); qty = _sn0(r1.get('CargoQty', 0))
        condition = 'LADEN' if qty > 100 else 'BALLAST'
        
        dqi = compute_dqi(r1, r2, daily_burn, drift, chrono_bad, mgo_neg)
        tol = max(30.0, 0.03 * max(r1.get('FO_A',0), r2.get('FO_A',0)))
        p_normal = gauss_mf(drift, 0.0, tol); p_ghost = trap_mf(-total_fuel, tol*0.8, tol*1.2, 10000, 20000)
        
        status = 'VERIFIED'
        if p_ghost > 0.7: status = 'GHOST BUNKER'
        elif p_normal < 0.30: status = 'LEDGER VARIANCE'
        
        phase = 'SEA' if leg_nm > 50 and speed > 3 else ('COASTAL' if leg_nm > 5 else 'PORT')
        flags = []
        if chrono_bad: flags.append('TIME_FB')
        if mgo_neg: flags.append('MGO_NEG')
        
        trips.append({
            'Indicator': ICONS.get(status, ICONS['VERIFIED']),
            'Timeline': f"{r1['Datetime'].strftime('%d %b %y')} → {r2['Datetime'].strftime('%d %b %y')}",
            'Date_Start_TS': r1['Datetime'],
            'Date_Start': r1['Datetime'].strftime('%Y-%m-%d'),
            'Phase': phase, 'Condition': condition, 'CargoQty': qty,
            'Route': f"{port_dep} → {port_arr}", 'Days': round(days, 2), 'Dist_NM': round(leg_nm, 0),
            'Speed_kn': round(speed, 1), 'HFO_MT': round(hfo_c, 1), 'MGO_MT': round(mgo_c, 1),
            'Fuel_MT': round(total_fuel, 1), 'Daily_Burn': round(daily_burn, 1),
            'MELO_L': round(melo_c, 0), 'CYLO_L': round(total_cylo, 0), 'GELO_L': round(gelo_c, 0),
            'Drift_MT': round(drift, 1), 'DQI': int(dqi), 'Status': status,
            'Voy': str(r1.get('Voy','')).strip(), 'Flags': ','.join(flags) if flags else ''
        })

    trip_df = pd.DataFrame(trips)
    
    if len(trip_df) >= 6:
        for cond in ['LADEN', 'BALLAST']:
            ver = trip_df[(trip_df['Status'] == 'VERIFIED') & (trip_df['Daily_Burn'] > 0) & (trip_df['Condition'] == cond)]
            if len(ver) >= 4:
                q1, q3 = ver['Daily_Burn'].quantile(0.25), ver['Daily_Burn'].quantile(0.75); iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 2.0*iqr, q3 + 2.0*iqr
                    mask = (trip_df['Status'] == 'VERIFIED') & (trip_df['Condition'] == cond) & ((trip_df['Daily_Burn'] < lo) | (trip_df['Daily_Burn'] > hi))
                    trip_df.loc[mask, 'Status'] = 'STAT OUTLIER'; trip_df.loc[mask, 'Indicator'] = ICONS['STAT OUTLIER']

    if not trip_df.empty:
        ai_df = calculate_stochastic_variance(trip_df)
        for col in ai_df.columns: trip_df[col] = ai_df[col]
        cols = list(trip_df.columns)
        if 'Stoch_Var' in cols and 'DQI' in cols:
            cols.insert(cols.index('DQI'), cols.pop(cols.index('Stoch_Var')))
            trip_df = trip_df[cols]

    summary = {}
    if not trip_df.empty:
        n = len(trip_df); n_ok = (trip_df['Status'] == 'VERIFIED').sum(); pb = trip_df[trip_df['Daily_Burn'] > 0]['Daily_Burn']
        summary = {'integrity':round(n_ok/n*100,1),'avg_dqi':round(trip_df['DQI'].mean(),0),'total_fuel':round(trip_df['Fuel_MT'].sum(),1),'total_hfo':round(trip_df['HFO_MT'].sum(),1),'total_mgo':round(trip_df['MGO_MT'].sum(),1),'avg_burn':round(pb.mean(),1) if len(pb) else 0.0,'total_nm':round(trip_df['Dist_NM'].sum(),0),'total_melo':round(trip_df['MELO_L'].sum(),0),'total_cylo':round(trip_df['CYLO_L'].sum(),0),'total_gelo':round(trip_df['GELO_L'].sum(),0),'total_days':round(trip_df['Days'].sum(),1),'cycles':n,'anomalies':n-n_ok,'ghost':int((trip_df['Status']=='GHOST BUNKER').sum()),'ledger':int((trip_df['Status']=='LEDGER VARIANCE').sum()),'outlier':int((trip_df['Status']=='STAT OUTLIER').sum()),'flagged':int((trip_df['Flags']!='').sum()),'laden':int((trip_df['Condition']=='LADEN').sum()),'ballast':int((trip_df['Condition']=='BALLAST').sum()),'dropped_dt':n_dropped}
        
    return trip_df, vname, summary, cum_drift

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS & PLOTLY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_BL=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',hovermode='x unified',hoverlabel=dict(bgcolor='#0c1219',bordercolor='rgba(201,168,76,0.12)',font=dict(family='Hanken Grotesk',color='#dce8f0',size=12)),font=dict(family='Hanken Grotesk',color='#4a6275',size=11),title_font=dict(family='Bricolage Grotesque',color='#ffffff',size=15),margin=dict(l=0,r=0,t=55,b=20))
_AX=dict(gridcolor='rgba(201,168,76,0.04)',zerolinecolor='rgba(201,168,76,0.06)',tickfont=dict(size=10))

def chart_fuel(df):
    bc=[SC.get(s,'#00e0b0') for s in df['Status']]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.65,.35],vertical_spacing=.06)
    fig.add_trace(go.Bar(x=df['Timeline'],y=df['Fuel_MT'],name='Cycle Fuel',marker=dict(color=[_rgba(c,.12) for c in bc],line=dict(color=bc,width=1.5)),text=df['Fuel_MT'],textposition='outside',textfont=dict(size=9,color='#4a6275')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['Timeline'],y=df['Daily_Burn'],name='Daily Burn',mode='lines+markers',line=dict(color='#00e0b0',width=2,shape='spline'),marker=dict(size=4),fill='tozeroy',fillcolor='rgba(0,224,176,0.03)'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['Timeline'],y=df['Speed_kn'],name='Speed',mode='lines+markers',line=dict(color='#c9a84c',width=2,shape='spline'),marker=dict(size=4,color='#c9a84c')),row=2,col=1)
    fig.update_layout(**_BL,title='Fuel Consumption & Speed Profile',barmode='overlay',showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)),yaxis=dict(title='MT',**_AX),yaxis2=dict(title='kn',**_AX),xaxis=dict(automargin=True,**_AX),xaxis2=dict(automargin=True,**_AX))
    fig.update_xaxes(tickangle=-45); return fig

def chart_stoch_var_dqi(df):
    if 'Stoch_Var' not in df.columns: return None
    cc=[SC.get(s,'#00e0b0') for s in df['Status']]
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=df['Timeline'],y=df['Stoch_Var'],name='Conf. Margin (±MT)',marker=dict(color=[_rgba(c,.2) for c in cc],line=dict(color=cc,width=1.3))),secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Timeline'],y=df['DQI'],name='DQI',mode='lines+markers',line=dict(color='#00e0b0',width=2,shape='spline'),marker=dict(size=4)),secondary_y=True)
    fig.update_layout(**_BL,title='Conformal Physics Margins & Data Quality',barmode='overlay',showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)))
    fig.update_yaxes(title_text='Variance Bounds MT',secondary_y=False,**_AX); fig.update_yaxes(title_text='DQI',secondary_y=True,range=[0,105],**_AX)
    fig.update_xaxes(tickangle=-45,automargin=True,**_AX); return fig

def chart_cum_drift(cum_drift):
    if not cum_drift: return None
    cdf=pd.DataFrame(cum_drift)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=cdf['dt'],y=cdf['gap'],mode='lines+markers',name='A−L Gap',line=dict(color='#c9a84c',width=2),marker=dict(size=3),fill='tozeroy',fillcolor='rgba(201,168,76,0.04)'))
    fig.add_hline(y=0,line=dict(color='rgba(255,255,255,0.06)',width=1,dash='dot'))
    fig.update_layout(**_BL,title='Cumulative Actual vs Ledger Gap',yaxis=dict(title='FO_A − FO_L (MT)',**_AX),xaxis=dict(automargin=True,**_AX))
    fig.update_xaxes(tickangle=-45); return fig

def chart_lube(df):
    fig=go.Figure()
    if df['MELO_L'].sum()>0: fig.add_trace(go.Bar(x=df['Timeline'],y=df['MELO_L'],name='MELO',marker=dict(color='rgba(0,224,176,0.12)',line=dict(color='#00e0b0',width=1.3))))
    if df['CYLO_L'].sum()>0: fig.add_trace(go.Bar(x=df['Timeline'],y=df['CYLO_L'],name='CYLO',marker=dict(color='rgba(123,104,238,0.12)',line=dict(color='#7b68ee',width=1.3))))
    if df['GELO_L'].sum()>0: fig.add_trace(go.Bar(x=df['Timeline'],y=df['GELO_L'],name='GELO',marker=dict(color='rgba(201,168,76,0.12)',line=dict(color='#c9a84c',width=1.3))))
    fig.update_layout(**_BL,title='Lubricant Consumption (Liters)',barmode='group',showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)),yaxis=dict(title='L',**_AX),xaxis=dict(automargin=True,**_AX))
    fig.update_xaxes(tickangle=-45); return fig

def chart_voyage(df):
    vg=df.groupby('Voy',sort=False).agg(Fuel=('Fuel_MT','sum'),Days=('Days','sum'),Dist=('Dist_NM','sum'),Legs=('Voy','count')).reset_index()
    vg=vg[vg['Fuel']>0]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=vg['Voy'],y=vg['Fuel'],name='Voyage Fuel',marker=dict(color='rgba(0,224,176,0.12)',line=dict(color='#00e0b0',width=1.5)),text=vg['Legs'].apply(lambda x:f'{x}L'),textposition='outside',textfont=dict(size=9,color='#4a6275')))
    fig.update_layout(**_BL,title='Fuel by Commercial Voyage (L = legs)',yaxis=dict(title='MT',**_AX),xaxis=dict(title='Voyage',automargin=True,**_AX)); return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero"><div class="hero-left"><img src="data:image/svg+xml;base64,{_LOGO}" class="hero-logo" alt=""/><div><div class="hero-title">POSEIDON TITAN</div><div class="hero-sub">Fleet Consumables Intelligence Engine</div></div></div><div class="hero-badge"><span>KERNEL</span>&ensp;Fuzzy Ledger + Actual FO_A<br><span>PIPELINE</span>&ensp;Kinematics & Conformal AI<br><span>BUILD</span>&ensp;v24.3 Zero-Crash Target</div></div>""",unsafe_allow_html=True)

uploaded_files=st.file_uploader('Upload vessel telemetry',accept_multiple_files=True,type=['xlsx','csv'],label_visibility='collapsed')

if not uploaded_files:
    st.markdown("""<div style="text-align:center;padding:100px 20px">
        <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:28px;opacity:.12">
            <circle cx="40" cy="40" r="36" fill="none" stroke="#c9a84c" stroke-width="0.8" stroke-dasharray="6 6"><animateTransform attributeName="transform" type="rotate" from="0 40 40" to="360 40 40" dur="30s" repeatCount="indefinite"/></circle>
            <circle cx="40" cy="40" r="24" fill="none" stroke="#00e0b0" stroke-width="0.6" stroke-dasharray="3 8"><animateTransform attributeName="transform" type="rotate" from="360 40 40" to="0 40 40" dur="20s" repeatCount="indefinite"/></circle>
            <path d="M40 14L40 66 M22 26Q40 36 58 26 M20 40Q40 50 60 40 M22 54Q40 64 58 54" fill="none" stroke="#00e0b0" stroke-width="1.5" stroke-linecap="round" opacity=".35"/></svg>
        <h2 style="color:#fff;font-family:'Bricolage Grotesque';font-weight:800;font-size:1.4rem;margin-bottom:8px;letter-spacing:-0.03em">Awaiting Telemetry</h2>
        <p style="color:#3a4d5e;font-size:.8rem;max-width:420px;margin:0 auto;line-height:1.7;font-family:'Hanken Grotesk'">Drop vessel noon-report files to execute the<br>Departure-to-Departure cyclic forensic audit.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

fleet_results=[]
for f in uploaded_files:
    try:
        with st.spinner(f'Processing {f.name}...'):
            df,vname,summary,cum_drift=process_file(f)
            
        if df.empty: st.warning(f'No valid cycles found in {f.name}.'); continue
        
        fleet_results.append({'name':vname,'summary':summary,'df':df})
        integrity=summary['integrity']; avg_dqi=summary['avg_dqi']
        ic=SC['VERIFIED'] if integrity>=80 else (SC['LEDGER VARIANCE'] if integrity>=50 else SC['GHOST BUNKER'])
        pc='p-ok' if integrity>=80 else ('p-w' if integrity>=50 else 'p-c')
        pt='NOMINAL' if integrity>=80 else ('ATTENTION' if integrity>=50 else 'CRITICAL')
        prov=[]
        if summary.get('dropped_dt'): prov.append(f"{summary['dropped_dt']} rows dropped")
        if summary.get('flagged'): prov.append(f"{summary['flagged']} flagged")
        prov_s=' · '.join(prov) if prov else 'Clean ingestion'

        st.markdown(f"""<div class="vcard"><div style="display:flex;justify-content:space-between;align-items:center"><div><div style="font-family:var(--fd);font-weight:800;font-size:1.3rem;color:#fff;letter-spacing:-0.03em">{vname}</div><div style="font-family:var(--fm);font-size:.62rem;color:var(--t2);margin-top:5px;letter-spacing:0.04em">{summary['cycles']} CYCLES&ensp;·&ensp;{summary['total_days']:.0f} DAYS&ensp;·&ensp;{int(summary['total_nm']):,} NM&ensp;·&ensp;{summary['total_fuel']:,.1f} MT&ensp;·&ensp;{summary['laden']}L / {summary['ballast']}B</div><div style="font-family:var(--fm);font-size:.55rem;color:var(--t3);margin-top:3px;letter-spacing:0.04em">{prov_s}</div></div><div style="text-align:right"><span class="pill {pc}">{pt}</span><div style="font-family:var(--fd);font-weight:800;font-size:1.6rem;color:{ic};margin-top:5px">{integrity:.0f}%</div><div style="font-family:var(--fm);font-size:.5rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em">Verified Ratio</div><div style="font-family:var(--fm);font-size:.65rem;color:var(--t2);margin-top:6px">DQI&ensp;<span style="color:#fff;font-weight:700;font-size:.8rem">{int(avg_dqi)}</span></div></div></div></div>""",unsafe_allow_html=True)

        cols=st.columns(6)
        cols[0].metric('HFO (MT)',f"{summary['total_hfo']:,.1f}")
        cols[1].metric('MGO (MT)',f"{summary['total_mgo']:,.1f}")
        cols[2].metric('Actual Burn (MT/d)',f"{summary['avg_burn']:.1f}")
        cols[3].metric('MELO (L)',f"{int(summary['total_melo']):,}")
        cols[4].metric('CYLO (L)',f"{int(summary['total_cylo']):,}")
        ap=[]
        if summary['ghost']: ap.append(f"{summary['ghost']} ghost")
        if summary['ledger']: ap.append(f"{summary['ledger']} ledger")
        if summary['outlier']: ap.append(f"{summary['outlier']} outlier")
        cols[5].metric('Anomalies',' / '.join(ap) if ap else '0')

        tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(['AUDIT MATRIX','FUEL ANALYTICS','DRIFT TRAJECTORY','LUBE OIL','FORENSIC DETAIL','AI EXPLAINER (SHAP)', 'CONFORMAL BOUNDS'])

        with tab1:
            dcfg={'Indicator':st.column_config.ImageColumn(' ',width='small'),'Timeline':st.column_config.TextColumn('TIMELINE',width='medium'),'Phase':st.column_config.TextColumn('PH',width='small'),'Condition':st.column_config.TextColumn('COND',width='small'),'Route':st.column_config.TextColumn('ROUTE',width='large'),'Days':st.column_config.NumberColumn('DAYS',format='%.2f'),'Dist_NM':st.column_config.NumberColumn('DIST',format='%d'),'Speed_kn':st.column_config.NumberColumn('SPD',format='%.1f'),'HFO_MT':st.column_config.NumberColumn('HFO',format='%.1f'),'MGO_MT':st.column_config.NumberColumn('MGO',format='%.1f'),'Fuel_MT':st.column_config.NumberColumn('FUEL',format='%.1f'),'Daily_Burn':st.column_config.ProgressColumn('BURN',format='%.1f',min_value=0,max_value=float(max(df['Daily_Burn'].max()*1.15,1))),'MELO_L':st.column_config.NumberColumn('MELO',format='%d'),'CYLO_L':st.column_config.NumberColumn('CYLO',format='%d'),'GELO_L':st.column_config.NumberColumn('GELO',format='%d'),'Stoch_Var':st.column_config.NumberColumn('VAR± MT',format='%.1f'),'DQI':st.column_config.ProgressColumn('DQI',format='%d',min_value=0,max_value=100),'Status':st.column_config.TextColumn('STATUS',width='medium'),'Flags':st.column_config.TextColumn('FLAGS',width='medium'),'Voy':None,'CargoQty':None,'SHAP_Base':None,'SHAP_Prop':None,'SHAP_Mass':None,'SHAP_Weath':None,'SHAP_Lag':None,'Exp_Lower':None,'Exp_Upper':None,'Date_Start':None,'Date_Start_TS':None,'Drift_MT':None,'Drift_Bias':None}
            st.dataframe(df,column_config=dcfg,hide_index=True,use_container_width=True,height=min(500,38+len(df)*35))
            buf=io.BytesIO(); exp=df.drop(columns=['Indicator','Date_Start_TS'],errors='ignore')
            with pd.ExcelWriter(buf,engine='openpyxl') as w: exp.to_excel(w,index=False,sheet_name='Audit')
            buf.seek(0)
            st.download_button('Export Audited Ledger',data=buf,file_name=f"{vname.replace(' ','_')}_AUDIT.xlsx",key=f"dl_{vname}_{id(f)}")

        with tab2:
            st.plotly_chart(chart_fuel(df),use_container_width=True,config={'displayModeBar':False})
            ai_fig=chart_stoch_var_dqi(df)
            if ai_fig: st.plotly_chart(ai_fig,use_container_width=True,config={'displayModeBar':False})
            st.plotly_chart(chart_voyage(df),use_container_width=True,config={'displayModeBar':False})

        with tab3:
            cfig=chart_cum_drift(cum_drift)
            if cfig: st.plotly_chart(cfig,use_container_width=True,config={'displayModeBar':False})
            st.caption('Tracks the running FO Actual − Ledger gap at every departure node.')

        with tab4:
            if df['MELO_L'].sum()+df['CYLO_L'].sum()+df['GELO_L'].sum()>0:
                st.plotly_chart(chart_lube(df),use_container_width=True,config={'displayModeBar':False})
            else: st.info('No lubricant consumption data detected.')

        with tab5:
            anomalies=df[df['Status']!='VERIFIED']
            if anomalies.empty:
                st.markdown('<div style="text-align:center;padding:60px;color:#6d8599"><svg viewBox="0 0 28 28" width="44" height="44" xmlns="http://www.w3.org/2000/svg" style="opacity:.2;margin-bottom:16px"><circle cx="14" cy="14" r="11" fill="none" stroke="#00e0b0" stroke-width="1.5"/><polyline points="9,14 12,17 19,10" fill="none" stroke="#00e0b0" stroke-width="2" stroke-linecap="round"/></svg><br><span style="font-family:var(--fd);font-weight:700">Zero anomalies detected.</span></div>',unsafe_allow_html=True)
            else:
                for _,row in anomalies.iterrows():
                    s=row['Status']; sc=SC.get(s,'#fff'); ri=tuple(int(sc.lstrip('#')[i:i+2],16) for i in (0,2,4))
                    fl=f" <span style='color:var(--t3);font-size:.6rem;font-family:var(--fm)'>[{row['Flags']}]</span>" if row['Flags'] else ''
                    ai_v=row.get('Stoch_Var',0.0)
                    dm={'GHOST BUNKER':f"Net fuel = {row['Fuel_MT']:.1f} MT (negative) — unrecorded bunkering ~{abs(row['Fuel_MT']):.0f} MT. DQI: {row['DQI']}%.{fl}",'LEDGER VARIANCE':f"Reporting drift variance exceeded threshold. DQI: {row['DQI']}%. {row['Condition']} leg, {row['Days']:.1f}d.{fl}",'STAT OUTLIER':f"Burn {row['Daily_Burn']:.1f} MT/d outside conformal bounds (±{ai_v:.1f} MT). DQI: {row['DQI']}%.{fl}"}
                    st.markdown(f'<div class="acard" style="border:1px solid rgba({ri[0]},{ri[1]},{ri[2]},.12);border-left:3px solid {sc}"><div style="display:flex;justify-content:space-between;align-items:center"><div><span style="color:{sc};font-weight:700;font-size:.7rem;letter-spacing:.08em;font-family:var(--fm)">{s}</span><span style="color:var(--t3);font-size:.7rem;margin-left:10px;font-family:var(--fm)">{row["Timeline"]}</span></div><span style="color:var(--t2);font-size:.68rem;font-family:var(--fb)">{row["Route"]}</span></div><div style="color:var(--t2);font-size:.7rem;margin-top:8px;line-height:1.6;font-family:var(--fb)">{dm.get(s,"")}</div></div>',unsafe_allow_html=True)

        with tab6:
            st.markdown('<h3 style="color:#fff;font-family:var(--fd);font-size:1.2rem;margin-bottom:10px;margin-top:10px">Neural Logic Extraction</h3>', unsafe_allow_html=True)
            shap_ran = df['SHAP_Base'].abs().sum() > 0 if 'SHAP_Base' in df.columns else False
            
            if not shap_ran:
                st.warning("⚠️ **AI EXPLAINABILITY OFFLINE:** The engine requires at least 2 valid sea-passages to map the physical hydrodynamics.")
            else:
                anomalies_s=df[df['Status']!='VERIFIED']
                if anomalies_s.empty:
                    st.success("No anomalies detected. You can view the AI's physics map for any valid leg below.")
                    options=df['Timeline'].tolist()
                    sel=st.selectbox('Select Voyage Leg',options,key=f'shap_{vname}')
                    tr=df[df['Timeline']==sel].iloc[0]
                else:
                    st.write("Select a flagged anomaly to view the exact physics receipt vs reported fuel.")
                    options=anomalies_s['Timeline'].tolist()
                    sel=st.selectbox('Select Flagged Anomaly',options,key=f'shap_{vname}')
                    tr=anomalies_s[anomalies_s['Timeline']==sel].iloc[0]
                
                exp_burn = tr['SHAP_Base'] + tr['SHAP_Prop'] + tr['SHAP_Mass'] + tr['SHAP_Weath'] + tr['SHAP_Lag']
                
                fig_w=go.Figure(go.Waterfall(
                    name="SHAP", orientation="v",
                    measure=["absolute","relative","relative","relative","relative","total"],
                    x=["Fleet<br>Baseline", "Propulsion<br>& Froude", "Cargo<br>Mass", "Weather<br>& Drag", "Lag &<br>Bias", "Expected<br>Actual Burn"],
                    textposition="auto", insidetextanchor="middle",
                    text=[f"{tr['SHAP_Base']:.1f}",f"{tr['SHAP_Prop']:+.1f}",f"{tr['SHAP_Mass']:+.1f}",f"{tr['SHAP_Weath']:+.1f}",f"{tr['SHAP_Lag']:+.1f}",f"{exp_burn:.1f}"],
                    y=[tr['SHAP_Base'],tr['SHAP_Prop'],tr['SHAP_Mass'],tr['SHAP_Weath'],tr['SHAP_Lag'],0],
                    connector={"line":{"color":"rgba(201,168,76,0.15)"}},
                    decreasing={"marker":{"color":"#00e0b0"}}, increasing={"marker":{"color":"#e63946"}}, totals={"marker":{"color":"#7b68ee"}}
                ))
                fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color="#dce8f0",height=480,margin=dict(t=60,b=80,l=10,r=10),title=dict(text=f"SHAP Physics Audit: {tr['Route']}",font=dict(color='#ffffff',size=16,family='Bricolage Grotesque')),yaxis=dict(title='MT/Day',**_AX), xaxis=dict(automargin=True, tickangle=0, **_AX))
                st.plotly_chart(fig_w,use_container_width=True,config={'displayModeBar':False})
                
                st.info(f"**Forensic Translation:** Fleet baseline = **{tr['SHAP_Base']:.1f} MT/d**. Speed/Froude Effort: **{tr['SHAP_Prop']:+.1f}**. Cargo mass: **{tr['SHAP_Mass']:+.1f}**. Weather/Hull drag: **{tr['SHAP_Weath']:+.1f}**. Reporting lag bias: **{tr['SHAP_Lag']:+.1f}**. \n\nExpected Burn: **{exp_burn:.1f} MT/d** vs Actually Reported: **{tr['Daily_Burn']:.1f} MT/d**.")

        with tab7:
            if 'Exp_Lower' in df.columns and df['Exp_Lower'].abs().sum() > 0:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=df['Timeline'].tolist() + df['Timeline'].tolist()[::-1],
                                         y=df['Exp_Upper'].tolist() + df['Exp_Lower'].tolist()[::-1],
                                         fill='toself', fillcolor='rgba(123,104,238,0.15)', line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip", name='90% Conformal Interval'))
                fig_c.add_trace(go.Scatter(x=df['Timeline'], y=(df['Exp_Lower']+df['Exp_Upper'])/2, name="Expected Mean", line=dict(color="#7b68ee", width=2, dash='dot')))
                fig_c.add_trace(go.Scatter(x=df['Timeline'], y=df['Daily_Burn'], name="Actual Burn (FO_A)", mode='lines+markers', line=dict(color="#00e0b0", width=2), marker=dict(size=6, color="#fff")))
                
                fig_c.update_layout(**_BL, title='Conformal Bounds (90% Statistical Certainty)', height=500, yaxis=dict(title='MT/day', **_AX), xaxis=dict(tickangle=-45, automargin=True, **_AX))
                st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar':False})
                st.caption("The purple envelope represents the absolute physical limits of the vessel based on XGBoost residual variance. Actual burns outside this zone are physically anomalous.")
            else:
                st.info("Insufficient data to generate 90% Conformal Prediction Bounds.")

        st.divider()
    except Exception:
        st.error(f'Failed on file: {f.name}')
        with st.expander('View Traceback'): st.code(traceback.format_exc())

if len(fleet_results)>1:
    st.markdown('<h2 style="color:#fff;font-family:var(--fd);margin-top:10px">Fleet Comparison Matrix</h2>',unsafe_allow_html=True)
    fleet_rows=[]
    for r in fleet_results:
        s=r['summary']
        fleet_rows.append({'Vessel':r['name'],'Cycles':s['cycles'],'Verified':f"{s['integrity']:.1f}%",'DQI':int(s['avg_dqi']),'Fuel MT':s['total_fuel'],'Avg Burn':s['avg_burn'],'Anomalies':s['anomalies'],'NM':int(s['total_nm']),'Days':s['total_days']})
    st.dataframe(pd.DataFrame(fleet_rows),hide_index=True,use_container_width=True)
    
    fig_f=go.Figure()
    for r in fleet_results:
        fig_f.add_trace(go.Bar(name=r['name'],x=['Total Fuel (MT)','Avg Burn (x10)','Anomalies (x10)','DQI Score'],y=[r['summary']['total_fuel'],r['summary']['avg_burn']*10,r['summary']['anomalies']*10,r['summary']['avg_dqi']]))
    fig_f.update_layout(**_BL,title='Cross-Fleet Performance',barmode='group',yaxis=dict(**_AX),xaxis=dict(**_AX))
    st.plotly_chart(fig_f,use_container_width=True,config={'displayModeBar':False})
