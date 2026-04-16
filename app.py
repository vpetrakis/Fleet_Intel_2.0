import streamlit as st
import pandas as pd
import numpy as np
import re, io, math, traceback, base64, warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# TITAN CORE: v27.0 THE MASTERPIECE (Tri-State + Semantic Parser + Conformal AI)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    import shap
    HAS_ML = True
except ImportError:
    HAS_ML = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="POSEIDON TITAN", page_icon="⚓", layout="wide", initial_sidebar_state="collapsed")

_LOGO = base64.b64encode(b'<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#c9a84c"/><stop offset="50%" stop-color="#00e0b0"/><stop offset="100%" stop-color="#005f73"/></linearGradient><filter id="glow"><feGaussianBlur stdDeviation="1.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="24" cy="24" r="22" fill="none" stroke="url(#pg)" stroke-width="0.8" opacity=".3"><animate attributeName="r" values="22;23;22" dur="5s" repeatCount="indefinite"/></circle><circle cx="24" cy="24" r="16" fill="none" stroke="url(#pg)" stroke-width="0.5" opacity=".15" stroke-dasharray="3 5"><animateTransform attributeName="transform" type="rotate" from="0 24 24" to="360 24 24" dur="30s" repeatCount="indefinite"/></circle><g filter="url(#glow)"><path d="M24 6L24 42" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round" opacity=".6"/><path d="M12 16Q24 24 36 16" fill="none" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"><animate attributeName="d" values="M12 16Q24 24 36 16;M12 18Q24 22 36 18;M12 16Q24 24 36 16" dur="4s" repeatCount="indefinite"/></path><path d="M10 24Q24 32 38 24" fill="none" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"><animate attributeName="d" values="M10 24Q24 32 38 24;M10 26Q24 30 38 26;M10 24Q24 32 38 24" dur="4s" begin="0.5s" repeatCount="indefinite"/></path><path d="M12 32Q24 40 36 32" fill="none" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"><animate attributeName="d" values="M12 32Q24 40 36 32;M12 34Q24 38 36 34;M12 32Q24 40 36 32" dur="4s" begin="1s" repeatCount="indefinite"/></path></g></svg>').decode()

_CSS = '''<style>
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
.hero-badge{font-family:var(--fm);font-size:.55rem;color:var(--t3);text-align:right;line-height:2;letter-spacing:.06em}.hero-badge span{color:var(--acc);font-weight:600}
[data-testid="stFileUploader"]{background:var(--s1)!important;border:1px dashed var(--b2)!important;border-radius:var(--r)!important;padding:24px!important;transition:all .3s}
[data-testid="stFileUploader"]:hover{border-color:var(--acc2)!important;box-shadow:0 0 20px rgba(201,168,76,0.08)}
[data-testid="stFileUploader"] section{padding:0!important}
[data-testid="stFileUploader"] button{background:rgba(201,168,76,.08)!important;color:var(--acc2)!important;border:1px solid var(--b2)!important;border-radius:8px!important;font-weight:600!important;margin-top:10px!important}
div[data-testid="stMetric"]{background:linear-gradient(180deg,var(--s1),var(--s2))!important;border:1px solid var(--b1)!important;border-radius:var(--r);padding:18px 22px!important;position:relative;overflow:hidden;transition:border-color .3s}
div[data-testid="stMetric"]:hover{border-color:var(--b2)!important}
div[data-testid="stMetric"]::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--acc2),transparent);opacity:0;transition:opacity .3s}
div[data-testid="stMetric"]:hover::after{opacity:.3}
div[data-testid="stMetricLabel"]{font-size:.58rem!important;color:var(--t2)!important;text-transform:uppercase!important;letter-spacing:.14em!important;font-weight:600!important;font-family:var(--fm)!important; white-space: normal; word-wrap: break-word;}
div[data-testid="stMetricValue"]{font-size:1.5rem!important;font-weight:800!important;color:#fff!important;line-height:1.1!important;margin-top:6px!important;font-family:var(--fd)!important;letter-spacing:-.03em!important; white-space: normal; word-wrap: break-word;}
div[data-testid="stMetricValue"]>div{color:#fff!important}
.stTabs [data-baseweb="tab-list"]{gap:0;background:transparent;border-bottom:1px solid rgba(201,168,76,0.08); flex-wrap: wrap;}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;border-bottom:2px solid transparent;border-radius:0;padding:12px 18px;color:var(--t3);font-weight:600;font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;font-family:var(--fm);transition:all .3s}
.stTabs [data-baseweb="tab"]:hover{color:var(--t1)}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--acc)!important;border-bottom-color:var(--acc)!important}
.stTabs [data-baseweb="tab-highlight"]{display:none}
.stDataFrame{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--b1)!important}
.stDownloadButton>button{background:rgba(201,168,76,.06)!important;color:var(--acc2)!important;border:1px solid var(--b2)!important;border-radius:10px!important;font-weight:600!important;padding:10px 24px!important;transition:all .3s!important}
.stDownloadButton>button:hover{background:rgba(201,168,76,.12)!important;box-shadow:0 4px 30px rgba(201,168,76,0.08)!important;transform:translateY(-1px)!important}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.04),rgba(201,168,76,.08),rgba(201,168,76,0.04),transparent)!important;margin:32px 0!important}
.vcard{background:linear-gradient(165deg,var(--s1),rgba(0,95,115,0.04));border:1px solid var(--b1);border-radius:16px;padding:26px 32px;margin-bottom:20px;position:relative;overflow:hidden}
.vcard::before{content:'';position:absolute;top:0;left:5%;right:5%;height:1px;background:linear-gradient(90deg,transparent,var(--acc2),transparent);opacity:.2}
.acard{background:var(--s1);border-radius:10px;padding:16px 20px;margin-bottom:8px;border-left:3px solid var(--red); transition:transform .2s,box-shadow .2s}
.acard:hover{transform:translateX(3px);box-shadow:-3px 0 20px rgba(0,0,0,0.3)}
.pill{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.58rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-family:var(--fm)}
.p-ok{background:rgba(0,224,176,.06);color:var(--acc);border:1px solid rgba(0,224,176,.15)}
.p-w{background:rgba(212,168,67,.06);color:var(--amber);border:1px solid rgba(212,168,67,.15)}
.p-c{background:rgba(230,57,70,.06);color:var(--red);border:1px solid rgba(230,57,70,.15)}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--t3);border-radius:2px}
details{background:var(--s1)!important;border:1px solid var(--b1)!important;border-radius:var(--r)!important}
details summary{color:var(--t1)!important;font-weight:600!important}
div[data-baseweb="select"]>div{background:var(--s1);border-color:var(--b1);color:#fff}
</style>'''
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ICONS & BASE UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def _u(s): return f"data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}"
ICONS={"VERIFIED":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#00e0b0" stroke-width="1" opacity=".2"><animate attributeName="r" values="12;13;12" dur="3s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#061a14" stroke="#00e0b0" stroke-width="1.2" filter="url(#g)"/><polyline points="10,14.5 12.8,17 18,10.5" fill="none" stroke="#00e0b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),"GHOST BUNKER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g2"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="14" cy="14" r="12" fill="none" stroke="#e63946" stroke-width="1" stroke-dasharray="4 3"><animateTransform attributeName="transform" type="rotate" from="0 14 14" to="360 14 14" dur="8s" repeatCount="indefinite"/></circle><circle cx="14" cy="14" r="7.5" fill="#1a0508" stroke="#e63946" stroke-width="1.2" filter="url(#g2)"/><g stroke="#e63946" stroke-width="2" stroke-linecap="round"><line x1="11" y1="11" x2="17" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line><line x1="17" y1="11" x2="11" y2="17"><animate attributeName="opacity" values="1;.3;1" dur="1.2s" repeatCount="indefinite"/></line></g></svg>'),"LEDGER VARIANCE":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g3"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><polygon points="14,3 3,25 25,25" fill="none" stroke="#d4a843" stroke-width="1.2" stroke-linejoin="round" filter="url(#g3)"><animate attributeName="stroke-opacity" values="1;.3;1" dur="2s" repeatCount="indefinite"/></polygon><line x1="14" y1="11" x2="14" y2="18" stroke="#d4a843" stroke-width="2" stroke-linecap="round"/><circle cx="14" cy="21.5" r="1.2" fill="#d4a843"/></svg>'),"STAT OUTLIER":_u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><defs><filter id="g4"><feGaussianBlur stdDeviation="1" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><rect x="4" y="4" width="20" height="20" rx="5" fill="none" stroke="#7b68ee" stroke-width="1.2" filter="url(#g4)"><animate attributeName="stroke-dasharray" values="0,80;80,0;0,80" dur="4s" repeatCount="indefinite"/></rect><circle cx="14" cy="14" r="4.5" fill="#0e0a1e" stroke="#7b68ee" stroke-width="1.2"/><circle cx="14" cy="14" r="1.8" fill="#7b68ee"><animate attributeName="r" values="1.8;2.8;1.8" dur="2s" repeatCount="indefinite"/></circle></svg>')}
SC={"VERIFIED":"#00e0b0","GHOST BUNKER":"#e63946","LEDGER VARIANCE":"#d4a843","STAT OUTLIER":"#7b68ee"}
def _rgba(h,a):
    h=h.lstrip('#')
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

OPS_KW=['RDV','OPL','STRAIT','CANAL','SECTOR','ZONE','RV PT','RV POINT','PILOT','ANCH','ROADSTEAD','TRAFFIC','SEPARATION','PSTN','KUMKALE','GELIBOLU','TURKELI','GREAT BELT']
def _is_ops(n): return any(k in str(n).upper() for k in OPS_KW)

def gauss_mf(v,c,s):
    if s<=0: return 1.0 if v==c else 0.0
    return math.exp(-0.5*((v-c)/s)**2)
def trap_mf(v,a,b,c,d):
    if v<=a or v>=d: return 0.0
    if a<v<b: return (v-a)/(b-a)
    if b<=v<=c: return 1.0
    if c<v<d: return (d-v)/(d-c)
    return 0.0

def _sn(val):
    if val is None or pd.isna(val): return np.nan
    if isinstance(val,float): return val
    if isinstance(val,int): return float(val)
    s=str(val).strip()
    if s=='' or s.upper() in ('NAN','N/A','NA','-','NIL','NONE'): return np.nan
    s=re.sub(r'[^\d.\-]','',s)
    if not s or s in ('.','-','-.'): return np.nan
    try: return float(s)
    except: return np.nan
def _sn0(val):
    v=_sn(val); return 0.0 if np.isnan(v) else v

def _parse_dt(d_val,t_val):
    try:
        if isinstance(d_val,pd.Timestamp): d_str=d_val.strftime('%Y-%m-%d')
        elif pd.isna(d_val): return pd.NaT
        else:
            ds=str(d_val).strip()
            ds=re.sub(r'20224','2024',ds); ds=re.sub(r'20023','2023',ds)
            ds=re.sub(r'(\d+)\s+([A-Za-z]+)\.?\s+(\d{4})',lambda m:f"{m.group(3)}-{m.group(2)[:3]}-{m.group(1).zfill(2)}",ds)
            p=pd.to_datetime(ds,errors='coerce',format='mixed')
            if pd.isna(p): return pd.NaT
            d_str=p.strftime('%Y-%m-%d')
        if isinstance(t_val,pd.Timestamp): t_str=t_val.strftime('%H:%M')
        elif pd.isna(t_val): t_str='00:00'
        else:
            tr=re.sub(r'[HhLlTtUuCc\s]','',str(t_val).strip())
            m=re.match(r'^(\d{1,2}):(\d{2})',tr)
            if m: t_str=f"{m.group(1).zfill(2)}:{m.group(2)}"
            elif re.match(r'^\d{4}$',tr): t_str=f"{tr[:2]}:{tr[2:]}"
            elif re.match(r'^\d{3}$',tr): t_str=f"0{tr[0]}:{tr[1:]}"
            elif re.match(r'^\d{1,2}$',tr): t_str=f"{tr.zfill(2)}:00"
            else: t_str='00:00'
        return pd.to_datetime(f"{d_str} {t_str}",errors='coerce')
    except: return pd.NaT

ZERO_COLS=['DistLeg','TotalDist','TotalTime','Speed','Bunk_FO','Bunk_MGO','Bunk_MELO','Bunk_HSCYLO','Bunk_LSCYLO','Bunk_CYLO','Bunk_GELO','CargoQty']
ROB_COLS=['FO_A','FO_L','MGO_A','MGO_L','MELO_R','HSCYLO_R','LSCYLO_R','CYLO_R','GELO_R','FW_R']
TEXT_COLS=['Port','Voy','AD','CargoName']

# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC PARSER (No Hallucinations)
# ═══════════════════════════════════════════════════════════════════════════════
def _match_column(parent, child):
    p, c = parent.upper().strip(), child.upper().strip()
    pc, cc = p.replace(' ',''), c.replace(' ','')

    if not c or c == p:
        if re.search(r'VOY', pc) and 'DIST' not in pc: return 'Voy'
        if re.search(r'^PORT|LOCATION', pc) and 'SAID' not in pc and 'LOUIS' not in pc: return 'Port'
        if re.search(r'A/?D$|^AD$|ARR.*DEP|^STATUS$', pc): return 'AD'
        if re.search(r'^DATE', pc): return 'Date'
        if re.search(r'^TIME', pc) and 'TOTAL' not in pc: return 'Time'
        if re.search(r'DIST.*SINCE|DIST.*NOON|LEG.*DIST', pc): return 'DistLeg'
        if re.search(r'TOTAL.*DIST', pc): return 'TotalDist'
        if re.search(r'TOTAL.*TIME', pc): return 'TotalTime'
        if re.search(r'SPEED|AVR.*SPEED|GEN.*AVR', pc): return 'Speed'
        if re.search(r'CARGO.*NAME', pc): return 'CargoName'
        if re.search(r'^QUANTITY$|^QTY$', pc): return 'CargoQty'
        return None

    if re.search(r'BUNKER|RECEIV|LUBRICANT', p):
        if re.search(r'^FO$|^HFO|^VLSFO|^IFO|FUEL\s*OIL', cc): return 'Bunk_FO'
        if re.search(r'MGO|^GO$|^LSMGO', cc): return 'Bunk_MGO'
        if re.search(r'^MELO', cc): return 'Bunk_MELO'
        if re.search(r'HSCYLO|HS\s*CYL', cc): return 'Bunk_HSCYLO'
        if re.search(r'LSCYLO|LS\s*CYL', cc): return 'Bunk_LSCYLO'
        if re.search(r'^GELO', cc): return 'Bunk_GELO'
        if re.search(r'^CYLO$|^CYL\s*OIL', cc): return 'Bunk_CYLO'
        return None

    if re.search(r'^ROB|REMAIN', p):
        if re.search(r'FO\s*A|FO\s*ACT|FO\s*ARR', cc): return 'FO_A'
        if re.search(r'FO\s*L|FO\s*LED|FO\s*DEP', cc): return 'FO_L'
        if re.search(r'MGO\s*A|LSMGO\s*A', cc): return 'MGO_A'
        if re.search(r'MGO\s*L|LSMGO\s*L', cc): return 'MGO_L'
        if re.search(r'^MELO$', cc): return 'MELO_R'
        if re.search(r'HSCYLO|HS\s*CYL', cc): return 'HSCYLO_R'
        if re.search(r'LSCYLO|LS\s*CYL', cc): return 'LSCYLO_R'
        if re.search(r'^GELO', cc): return 'GELO_R'
        if re.search(r'^FW$|FRESH', cc): return 'FW_R'
        if re.search(r'^CYLO$', cc): return 'CYLO_R'
        return None

    return None

def semantic_parse(df_raw):
    kw = ['PORT','VOY','DATE','A/D','FO','ROB','DIST','BUNKER','SPEED','TIME','QUANTITY']
    best_row, best_sc = 0, 0
    for i in range(min(60, len(df_raw))):
        txt = ' '.join(str(x).upper() for x in df_raw.iloc[i].values if pd.notna(x))
        sc = sum(1 for k in kw if k in txt)
        if sc > best_sc: best_sc, best_row = sc, i

    ncols = len(df_raw.columns)
    h0 = [str(x).strip().upper() if pd.notna(x) else '' for x in df_raw.iloc[best_row].values]
    h1_idx = best_row + 1
    h1 = [str(x).strip().upper() if pd.notna(x) else '' for x in df_raw.iloc[h1_idx].values] if h1_idx < len(df_raw) else [''] * ncols

    last = ''
    for i in range(len(h0)):
        if h0[i]: last = h0[i]
        else: h0[i] = last

    col_map = {}
    mapped_names = set()
    for i in range(ncols):
        parent = h0[i] if i < len(h0) else ''
        child = h1[i] if i < len(h1) else ''
        canonical = _match_column(parent, child)
        if canonical and canonical not in mapped_names:
            col_map[i] = canonical
            mapped_names.add(canonical)

    data_start = best_row + 2
    df = df_raw.iloc[data_start:].copy().reset_index(drop=True)

    rename_dict = {}
    for col_idx, canon in col_map.items():
        if col_idx < len(df.columns): rename_dict[df.columns[col_idx]] = canon
    df = df.rename(columns=rename_dict)
    
    keep = [c for c in df.columns if c in mapped_names]
    df = df[keep]

    # Initialize structure safley
    for c in ZERO_COLS:
        if c not in df.columns: df[c] = 0.0
        else: df[c] = df[c].apply(_sn0)
    for c in ROB_COLS:
        if c not in df.columns: df[c] = np.nan
        else: df[c] = df[c].apply(_sn)
    for c in TEXT_COLS:
        if c not in df.columns: df[c] = ''
        else: df[c] = df[c].fillna('').astype(str).str.strip()
    for c in ['Date','Time']:
        if c not in df.columns: df[c] = ''

    return df

# ═══════════════════════════════════════════════════════════════════════════════
# DQI + CONFORMAL AI ENGINE (Tri-State Isolated)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dqi(r1, r2, daily_burn, drift, chrono_bad, mgo_neg):
    s={}
    rob_f=['FO_A','FO_L','MGO_A']
    s['rob'] = sum(1 for f in rob_f if not np.isnan(r1.get(f,np.nan)) and not np.isnan(r2.get(f,np.nan)))/len(rob_f)
    tol = max(30.0, 0.03 * max(r1.get('FO_A',0) or 0, r2.get('FO_A',0) or 0))
    s['drift'] = gauss_mf(drift, 0.0, tol)
    if daily_burn > 0: s['burn'] = gauss_mf(daily_burn, 30.0, 25.0)
    elif daily_burn == 0: s['burn'] = 0.5
    else: s['burn'] = 0.1
    s['chrono'] = 0.3 if chrono_bad else 1.0
    s['mgo'] = 0.3 if mgo_neg else 1.0
    w = {'rob':0.30,'drift':0.30,'burn':0.20,'chrono':0.10,'mgo':0.10}
    log_sum = sum(w[k]*math.log(max(v,0.001)) for k,v in s.items())
    return min(100, max(0, round(math.exp(log_sum)*100, 0)))

def compute_ai_diagnostics(trip_df):
    zeros=pd.DataFrame({'AI_Exp':[np.nan]*len(trip_df), 'Stoch_Var':[np.nan]*len(trip_df),
        'SHAP_Base':[np.nan]*len(trip_df), 'SHAP_Propulsion':[np.nan]*len(trip_df),
        'SHAP_Mass':[np.nan]*len(trip_df), 'SHAP_Kinematics':[np.nan]*len(trip_df),
        'SHAP_Season':[np.nan]*len(trip_df)}, index=trip_df.index)
    try:
        if not HAS_ML or trip_df.empty: return zeros
        
        # EXCLUSIVE ISOLATION: Model only trains and evaluates pure Sea Legs
        sea_mask = (trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Speed_kn'] > 2.0)
        if sea_mask.sum() < 2: return zeros
        
        ml = trip_df.loc[sea_mask].copy()
        
        # Enhanced Kinematic Feature Matrix
        ml['True_Mass'] = ml['CargoQty'].fillna(0) + ml['FO_A_Start'].fillna(0) + ml.get('MGO_A', 0)
        ml['SOG'] = ml['Dist_NM'] / np.maximum(ml['Days']*24, 0.1)
        ml['Kin_Delta'] = (ml['Speed_kn'] - ml['SOG']).clip(-3.0, 3.0)
        ml['Accel_Penalty'] = ml['Speed_kn'].diff().fillna(0.0).clip(-2, 2)
        ml['Speed_Cubed'] = ml['Speed_kn']**3
        ml['Season_Sin'] = np.sin(2*np.pi*ml['Date_Start_TS'].dt.month.fillna(6)/12.0)
        ml['Season_Cos'] = np.cos(2*np.pi*ml['Date_Start_TS'].dt.month.fillna(6)/12.0)
        
        features = ['Speed_kn', 'Speed_Cubed', 'True_Mass', 'Kin_Delta', 'Accel_Penalty', 'Season_Sin', 'Season_Cos']
        ml[features] = ml[features].fillna(0.0)
        
        # Dual Model (Mean & Variance) WITH Duration Weighting
        X_train = ml[features]
        y_train = ml['Phys_Burn']
        weights = ml['Days'] # 20-day ocean crossing pulls 20x harder than a 1-day hop

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.06, random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        
        train_preds = model.predict(X_train)
        residuals = np.abs(y_train - train_preds)
        
        var_model = XGBRegressor(n_estimators=40, max_depth=2, learning_rate=0.05, random_state=42)
        var_model.fit(X_train, residuals, sample_weight=weights)
        
        preds = model.predict(X_train)
        exp_var = var_model.predict(X_train)
        stoch_margin = np.maximum(exp_var * 1.645, 0.5) 
        
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_train)
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray): base_val = base_val[0]
        
        # Re-inject safely
        zeros.loc[sea_mask, 'AI_Exp'] = preds.round(1)
        zeros.loc[sea_mask, 'Stoch_Var'] = stoch_margin.round(1)
        zeros.loc[sea_mask, 'SHAP_Base'] = base_val
        zeros.loc[sea_mask, 'SHAP_Propulsion'] = sv[:,0] + sv[:,1] # Speed + Cubed
        zeros.loc[sea_mask, 'SHAP_Mass'] = sv[:,2]
        zeros.loc[sea_mask, 'SHAP_Kinematics'] = sv[:,3] + sv[:,4] # Delta + Accel
        zeros.loc[sea_mask, 'SHAP_Season'] = sv[:,5] + sv[:,6]
        
        return zeros
    except Exception: return zeros

# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION: THE FORENSIC AD-TO-AD STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def master_ingest(uploaded_file):
    vn_raw = re.sub(r'\.[^.]+$', '', uploaded_file.name).strip()
    vname = re.sub(r'[_\-]+', ' ', vn_raw).upper()
    uploaded_file.seek(0)
    
    if uploaded_file.name.lower().endswith('.xlsx'): df_raw = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
    else: df_raw = pd.read_csv(io.StringIO(uploaded_file.read().decode('latin-1', errors='replace')), sep=None, engine='python', header=None, on_bad_lines='skip')
        
    if df_raw.empty or len(df_raw) < 4: return pd.DataFrame(), vname, {}, []

    df = semantic_parse(df_raw)

    df['Datetime'] = df.apply(lambda r: _parse_dt(r.get('Date'), r.get('Time')), axis=1)
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    
    def _cad(v):
        v = str(v).strip().upper().replace(' ','')
        if v in ['D', 'DEP', 'SBE', 'FAOP']: return 'D'
        if v.startswith('A') and 'D' not in v: return 'A'
        return v
    df['AD'] = df['AD'].apply(_cad)
    
    ad_events = df[df['AD'].isin(['A', 'D'])].copy()
    if len(ad_events) < 2: return pd.DataFrame(), vname, {}, []
    
    trips = []
    cum_drift = []
    
    for i in range(len(ad_events)-1):
        r1 = ad_events.iloc[i]
        r2 = ad_events.iloc[i+1]
        idx1, idx2 = r1.name, r2.name
        
        status = 'VERIFIED'
        flags = []
        phys_burn, log_burn, drift, days = np.nan, np.nan, np.nan, 0.0
        
        # 1. Strict State Verification
        if r1['AD'] == 'D' and r2['AD'] == 'A': phase = 'SEA'
        elif r1['AD'] == 'A' and r2['AD'] == 'D': phase = 'PORT'
        else: 
            phase = 'QUARANTINE_SEQ'
            status = 'CHAIN BREAK'
            flags.append(f"Invalid Seq: {r1['AD']} → {r2['AD']}")
            
        # 2. Time Verification
        if status == 'VERIFIED':
            days = (r2['Datetime'] - r1['Datetime']).total_seconds() / 86400.0
            if days <= 0:
                status = 'QUARANTINE_TIME'
                flags.append("Negative Time Delta")
                
        # 3. Physical ROB Verification (No Guessing)
        start_rob = _sn(r1.get('FO_A'))
        end_rob = _sn(r2.get('FO_A'))
        if status == 'VERIFIED' and (pd.isna(start_rob) or pd.isna(end_rob)):
            status = 'QUARANTINE_ROB'
            flags.append("Missing Physical Tank Sounding")
        
        # Cumulative Drift Tracking (Only on valid departures)
        if r1['AD'] == 'D' and not pd.isna(start_rob):
            fol = _sn(r1.get('FO_L'))
            cum_drift.append({'dt': r1['Datetime'], 'gap': start_rob - (fol if not pd.isna(fol) else start_rob), 'port': str(r1.get('Port',''))[:20]})
        
        window = df.loc[idx1+1:idx2]
        
        if phase == 'PORT':
            bfo = _sn0(df.loc[idx1:idx2, 'Bunk_FO'].sum())
            bmelo = _sn0(df.loc[idx1:idx2, 'Bunk_MELO'].sum())
            bcylo = _sn0(df.loc[idx1:idx2, 'Bunk_HSCYLO'].sum() + df.loc[idx1:idx2, 'Bunk_LSCYLO'].sum() + df.loc[idx1:idx2, 'Bunk_CYLO'].sum())
            bgelo = _sn0(df.loc[idx1:idx2, 'Bunk_GELO'].sum())
        else:
            bfo = _sn0(window['Bunk_FO'].sum())
            bmelo = _sn0(window['Bunk_MELO'].sum())
            bcylo = _sn0(window['Bunk_HSCYLO'].sum() + window['Bunk_LSCYLO'].sum() + window['Bunk_CYLO'].sum())
            bgelo = _sn0(window['Bunk_GELO'].sum())
        
        dist = _sn0(window['DistLeg'].sum())
        if dist <= 0 and phase == 'SEA': dist = max(0, _sn0(r2.get('TotalDist')) - _sn0(r1.get('TotalDist')))
        
        speed = window['Speed'].replace(0, np.nan).mean() if not window['Speed'].empty else np.nan
        if pd.isna(speed): speed = dist / (days * 24.0) if days > 0 else 0.0
        
        melo_c = max(0, (_sn0(r1.get('MELO_R')) - _sn0(r2.get('MELO_R'))) + bmelo)
        cylo_c = max(0, (_sn0(r1.get('HSCYLO_R')) + _sn0(r1.get('LSCYLO_R')) + _sn0(r1.get('CYLO_R'))) - (_sn0(r2.get('HSCYLO_R')) + _sn0(r2.get('LSCYLO_R')) + _sn0(r2.get('CYLO_R'))) + bcylo)
        gelo_c = max(0, (_sn0(r1.get('GELO_R')) - _sn0(r2.get('GELO_R'))) + bgelo)
        
        dqi = 0
        if status == 'VERIFIED':
            phys_burn = (start_rob - end_rob) + bfo
            
            log_start = _sn(r1.get('FO_L')) if not pd.isna(_sn(r1.get('FO_L'))) else start_rob
            log_end = _sn(r2.get('FO_L')) if not pd.isna(_sn(r2.get('FO_L'))) else end_rob
            log_burn = (log_start - log_end) + bfo
            
            drift = phys_burn - log_burn
            daily_burn = phys_burn / days if days > 0 else 0.0
            
            dqi = compute_dqi(r1, r2, daily_burn, drift, days<=0, False)
            
            if phys_burn < -2.0:
                status = 'GHOST BUNKER'
                flags.append("Mass Imbalance: End ROB > Start + Bunkers")
                
        route = f"{str(r1.get('Port',''))[:15]} → {str(r2.get('Port',''))[:15]}" if phase == 'SEA' else f"Port Idle: {str(r1.get('Port',''))[:15]}"
        qty = _sn0(r1.get('CargoQty', 0))
        
        trips.append({
            'Indicator': ICONS.get(status, ICONS['VERIFIED']) if 'QUARANTINE' not in status else '⛔',
            'Timeline': f"{r1['Datetime'].strftime('%d %b %y')} → {r2['Datetime'].strftime('%d %b %y')}",
            'Date_Start_TS': r1['Datetime'],
            'Phase': phase,
            'Condition': 'LADEN' if qty > 100 else 'BALLAST',
            'Voy': str(r1.get('Voy','')).strip(),
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
            'MELO_L': round(melo_c, 0),
            'CYLO_L': round(cylo_c, 0),
            'GELO_L': round(gelo_c, 0),
            'DQI': int(dqi),
            'Status': status,
            'Flags': ', '.join(flags) if flags else ''
        })

    trip_df = pd.DataFrame(trips)
    
    # Statistical Outliers (Only applied to pure sea kinematics)
    if len(trip_df) >= 4:
        for cond in ['LADEN', 'BALLAST']:
            ver = trip_df[(trip_df['Status'] == 'VERIFIED') & (trip_df['Phase'] == 'SEA') & (trip_df['Phys_Burn'] > 0) & (trip_df['Condition'] == cond)]
            if len(ver) >= 4:
                q1, q3 = ver['Daily_Burn'].quantile(0.25), ver['Daily_Burn'].quantile(0.75); iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 2.0*iqr, q3 + 2.0*iqr
                    mask = (trip_df['Status'] == 'VERIFIED') & (trip_df['Phase'] == 'SEA') & (trip_df['Condition'] == cond) & ((trip_df['Daily_Burn'] < lo) | (trip_df['Daily_Burn'] > hi))
                    trip_df.loc[mask, 'Status'] = 'STAT OUTLIER'; trip_df.loc[mask, 'Indicator'] = ICONS['STAT OUTLIER']

    # Inject AI Physics Bounds
    if not trip_df.empty:
        ai_df = compute_ai_diagnostics(trip_df)
        for col in ai_df.columns: trip_df[col] = ai_df[col]

    summary = {}
    if not trip_df.empty:
        verified_sea = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Phys_Burn'] > 0)]['Phys_Burn']
        verified_sea_days = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED') & (trip_df['Phys_Burn'] > 0)]['Days']
        avg_sea = verified_sea.sum() / verified_sea_days.sum() if verified_sea_days.sum() > 0 else 0.0
        
        quarantined = len(trip_df[trip_df['Status'].str.contains('QUARANTINE')])
        
        summary = {
            'integrity': round((len(trip_df) - quarantined) / len(trip_df) * 100, 1),
            'avg_dqi': round(trip_df['DQI'].mean(), 0),
            'total_fuel': round(trip_df['Phys_Burn'].sum(skipna=True), 1),
            'avg_sea_burn': round(avg_sea, 1),
            'total_nm': round(trip_df['Dist_NM'].sum(), 0),
            'total_melo': round(trip_df['MELO_L'].sum(), 0),
            'total_cylo': round(trip_df['CYLO_L'].sum(), 0),
            'total_days': round(trip_df['Days'].sum(), 1),
            'cycles': len(trip_df),
            'quarantined': quarantined,
            'anomalies': len(trip_df[trip_df['Status'].isin(['GHOST BUNKER', 'STAT OUTLIER'])])
        }
        
    return trip_df, vname, summary, cum_drift

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS & VISUALS
# ═══════════════════════════════════════════════════════════════════════════════
_BL=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',hovermode='x unified',hoverlabel=dict(bgcolor='#0c1219',bordercolor='rgba(201,168,76,0.12)',font=dict(family='Hanken Grotesk',color='#dce8f0',size=12)),font=dict(family='Hanken Grotesk',color='#4a6275',size=11),title_font=dict(family='Bricolage Grotesque',color='#ffffff',size=15),margin=dict(l=0,r=0,t=55,b=20))
_AX=dict(gridcolor='rgba(201,168,76,0.04)',zerolinecolor='rgba(201,168,76,0.06)',tickfont=dict(size=10))

def chart_fuel(df):
    sea = df[(df['Phase'] == 'SEA') & (~df['Status'].str.contains('QUARANTINE'))]
    port = df[(df['Phase'] == 'PORT') & (~df['Status'].str.contains('QUARANTINE'))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08)
    
    if not sea.empty: fig.add_trace(go.Bar(x=sea['Timeline'], y=sea['Phys_Burn'], name='Sea Fuel', marker_color='rgba(0,224,176,0.15)', marker_line_color='#00e0b0', marker_line_width=1.5), row=1, col=1)
    if not port.empty: fig.add_trace(go.Bar(x=port['Timeline'], y=port['Phys_Burn'], name='Port Fuel', marker_color='rgba(123,104,238,0.15)', marker_line_color='#7b68ee', marker_line_width=1.5), row=1, col=1)
    if not sea.empty: fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Daily_Burn'], name='Sea MT/day', mode='lines+markers', line=dict(color='#00e0b0', width=2, shape='spline')), row=1, col=1)
    if not sea.empty: fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Speed_kn'], name='Sea Speed', mode='lines+markers', line=dict(color='#c9a84c', width=2, shape='spline')), row=2, col=1)
    
    fig.update_layout(**_BL, title='Tri-State Fuel Consumption & Sea Speed Profile', barmode='group', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(tickangle=-45, automargin=True, **_AX); fig.update_yaxes(**_AX)
    return fig

def chart_cum_drift(cum_drift):
    if not cum_drift: return None
    cdf=pd.DataFrame(cum_drift)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=cdf['dt'],y=cdf['gap'],mode='lines+markers',name='A−L Gap',line=dict(color='#c9a84c',width=2),marker=dict(size=3),fill='tozeroy',fillcolor='rgba(201,168,76,0.04)'))
    fig.add_hline(y=0,line=dict(color='rgba(255,255,255,0.06)',width=1,dash='dot'))
    fig.update_layout(**_BL,title='Cumulative Physical vs Logged Mass Drift',yaxis=dict(title='FO_A − FO_L (MT)',**_AX),xaxis=dict(automargin=True, **_AX))
    fig.update_xaxes(tickangle=-45); return fig

def chart_lube(df):
    fig = go.Figure()
    if df['MELO_L'].sum() > 0: fig.add_trace(go.Bar(x=df['Timeline'], y=df['MELO_L'], name='MELO', marker_color='rgba(0,224,176,0.12)', marker_line_color='#00e0b0', marker_line_width=1.3))
    if df['CYLO_L'].sum() > 0: fig.add_trace(go.Bar(x=df['Timeline'], y=df['CYLO_L'], name='CYLO', marker_color='rgba(123,104,238,0.12)', marker_line_color='#7b68ee', marker_line_width=1.3))
    if df['GELO_L'].sum() > 0: fig.add_trace(go.Bar(x=df['Timeline'], y=df['GELO_L'], name='GELO', marker_color='rgba(201,168,76,0.12)', marker_line_color='#c9a84c', marker_line_width=1.3))
    fig.update_layout(**_BL, title='Lubricant Consumption (Liters)', barmode='group', showlegend=True, yaxis=dict(title='L', **_AX), xaxis=dict(automargin=True, **_AX))
    fig.update_xaxes(tickangle=-45); return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI (Epistemological Segregation)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero"><div class="hero-left"><img src="data:image/svg+xml;base64,{_LOGO}" class="hero-logo" alt=""/><div><div class="hero-title">POSEIDON TITAN</div><div class="hero-sub">Forensic Accounting & Intelligence Engine</div></div></div><div class="hero-badge"><span>KERNEL</span>&ensp;Zero-Tolerance State Machine<br><span>PIPELINE</span>&ensp;Duration-Weighted AI<br><span>BUILD</span>&ensp;v27.0 Masterpiece</div></div>""",unsafe_allow_html=True)

uploaded_files=st.file_uploader('Upload vessel telemetry',accept_multiple_files=True,type=['xlsx','csv'],label_visibility='collapsed')

if not uploaded_files:
    st.markdown("""<div style="text-align:center;padding:100px 20px">
        <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:28px;opacity:.12"><circle cx="40" cy="40" r="36" fill="none" stroke="#c9a84c" stroke-width="0.8" stroke-dasharray="6 6"><animateTransform attributeName="transform" type="rotate" from="0 40 40" to="360 40 40" dur="30s" repeatCount="indefinite"/></circle><circle cx="40" cy="40" r="24" fill="none" stroke="#00e0b0" stroke-width="0.6" stroke-dasharray="3 8"><animateTransform attributeName="transform" type="rotate" from="360 40 40" to="0 40 40" dur="20s" repeatCount="indefinite"/></circle><path d="M40 14L40 66 M22 26Q40 36 58 26 M20 40Q40 50 60 40 M22 54Q40 64 58 54" fill="none" stroke="#00e0b0" stroke-width="1.5" stroke-linecap="round" opacity=".35"/></svg>
        <h2 style="color:#fff;font-family:'Bricolage Grotesque';font-weight:800;font-size:1.4rem;margin-bottom:8px">Awaiting Telemetry</h2>
        <p style="color:#3a4d5e;font-size:.8rem;max-width:420px;margin:0 auto;line-height:1.7;font-family:'Hanken Grotesk'">Drop vessel noon-report files to execute the<br>Zero-Tolerance State Machine & Physics isolation.</p></div>""",unsafe_allow_html=True)
    st.stop()

fleet_results=[]
for f in uploaded_files:
    try:
        with st.spinner(f'Auditing {f.name}...'):
            df, vname, summary, cum_drift = master_ingest(f)
            
        if df.empty: st.warning(f'No valid events found in {f.name}. Check the template.'); continue
        fleet_results.append({'name':vname,'summary':summary,'df':df})
        
        integrity = summary['integrity']
        ic = SC['VERIFIED'] if integrity >= 80 else (SC['LEDGER VARIANCE'] if integrity >= 50 else SC['GHOST BUNKER'])
        pc = 'p-ok' if integrity >= 80 else ('p-w' if integrity >= 50 else 'p-c')
        pt = 'NOMINAL' if integrity >= 80 else ('ATTENTION' if integrity >= 50 else 'CRITICAL')

        st.markdown(f"""<div class="vcard"><div style="display:flex;justify-content:space-between;align-items:center"><div><div style="font-family:var(--fd);font-weight:800;font-size:1.3rem;color:#fff;letter-spacing:-0.03em">{vname}</div><div style="font-family:var(--fm);font-size:.62rem;color:var(--t2);margin-top:5px;letter-spacing:0.04em">{summary['cycles']} LEGS&ensp;·&ensp;{summary['total_days']:.0f} DAYS&ensp;·&ensp;{int(summary['total_nm']):,} NM</div><div style="font-family:var(--fm);font-size:.55rem;color:var(--t3);margin-top:3px;letter-spacing:0.04em">Deterministic Core Active</div></div><div style="text-align:right"><span class="pill {pc}">{pt}</span><div style="font-family:var(--fd);font-weight:800;font-size:1.6rem;color:{ic};margin-top:5px">{integrity:.0f}%</div><div style="font-family:var(--fm);font-size:.5rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em">Audit Integrity</div></div></div></div>""",unsafe_allow_html=True)

        cols=st.columns(6)
        cols[0].metric('Verified Fuel (MT)', f"{summary['total_fuel']:,.1f}")
        cols[1].metric('Avg Sea Burn (MT/d)', f"{summary['avg_sea_burn']:.1f}")
        cols[2].metric('Total MELO (L)', f"{int(summary['total_melo']):,}")
        cols[3].metric('Total CYLO (L)', f"{int(summary['total_cylo']):,}")
        cols[4].metric('Mass Anomalies', f"{summary['anomalies']}")
        cols[5].metric('Quarantined Legs', f"{summary['quarantined']}")

        # 6 TABS: Segregating Accounting, Commercial, and AI
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['IMMUTABLE LEDGER', 'COMMERCIAL P&L', 'LUBE & DRIFT', 'AI DIGITAL TWIN', 'SHAP EXPLAINER', 'QUARANTINE LOG'])

        with tab1:
            st.markdown('<span style="font-family:var(--fm); font-size:0.7rem; color:var(--acc)">[START ROB] + [BUNKERS] - [END ROB] = [PHYSICAL BURN]</span>', unsafe_allow_html=True)
            dcfg={'Indicator':st.column_config.ImageColumn(' ',width='small'),'Timeline':st.column_config.TextColumn('TIMELINE',width='medium'),'Phase':st.column_config.TextColumn('LEG',width='small'),'Days':st.column_config.NumberColumn('DAYS',format='%.2f'),'Speed_kn':st.column_config.NumberColumn('SPD',format='%.1f'),'FO_A_Start':st.column_config.NumberColumn('START ROB',format='%.1f'),'Bunk_FO':st.column_config.NumberColumn('+ BUNKERS',format='%.1f'),'FO_A_End':st.column_config.NumberColumn('- END ROB',format='%.1f'),'Phys_Burn':st.column_config.NumberColumn('= PHYS BURN',format='%.1f'),'Log_Burn':st.column_config.NumberColumn('LOG BURN',format='%.1f'),'DQI':st.column_config.ProgressColumn('DQI',format='%d',min_value=0,max_value=100),'Daily_Burn':st.column_config.NumberColumn('MT/DAY',format='%.1f'),'Status':st.column_config.TextColumn('STATUS',width='medium'),'Route':None,'CargoQty':None,'Condition':None,'Date_Start_TS':None,'Flags':None,'AI_Exp':None,'Stoch_Var':None,'SHAP_Base':None,'SHAP_Prop':None,'SHAP_Mass':None,'SHAP_Weath':None,'Exp_Lower':None,'Exp_Upper':None,'Dist_NM':None,'Voy':None,'Drift_MT':None, 'SHAP_Kinematics':None, 'SHAP_Season':None,'SHAP_Propulsion':None, 'MELO_L':None, 'CYLO_L':None, 'GELO_L':None}
            st.dataframe(df,column_config=dcfg,hide_index=True,use_container_width=True,height=min(500,38+len(df)*35))
            buf=io.BytesIO(); exp=df.drop(columns=['Indicator','Date_Start_TS'],errors='ignore')
            with pd.ExcelWriter(buf,engine='openpyxl') as w: exp.to_excel(w,index=False,sheet_name='Audit')
            buf.seek(0)
            st.download_button('Export Tri-State Ledger',data=buf,file_name=f"{vname.replace(' ','_')}_LEDGER.xlsx",key=f"dl_{vname}")

        with tab2:
            st.markdown('<h3 style="color:#fff;font-family:var(--fd);font-size:1.2rem;margin-bottom:10px;margin-top:10px">Commercial Voyage Matrix</h3>', unsafe_allow_html=True)
            voy_df = df[~df['Status'].str.contains('QUARANTINE')]
            if voy_df.empty:
                st.info("Insufficient valid data to generate Commercial Matrix.")
            else:
                vg = voy_df.groupby('Voy', dropna=False).agg(
                    Total_Fuel=('Phys_Burn', 'sum'),
                    Sea_Days=('Days', lambda x: x[voy_df['Phase'] == 'SEA'].sum()),
                    Port_Days=('Days', lambda x: x[voy_df['Phase'] == 'PORT'].sum()),
                    Sea_Fuel=('Phys_Burn', lambda x: x[voy_df['Phase'] == 'SEA'].sum()),
                    Port_Fuel=('Phys_Burn', lambda x: x[voy_df['Phase'] == 'PORT'].sum()),
                    Bunkers=('Bunk_FO', 'sum'),
                    Dist=('Dist_NM', 'sum'),
                    Avg_Sea_Spd=('Speed_kn', lambda x: x[voy_df['Phase'] == 'SEA'].mean())
                ).reset_index()
                vg['Avg_Sea_MT_Day'] = np.where(vg['Sea_Days'] > 0, vg['Sea_Fuel'] / vg['Sea_Days'], 0.0)
                
                v_cfg = {'Voy':st.column_config.TextColumn('VOYAGE NO.'),'Total_Fuel':st.column_config.NumberColumn('TOTAL FUEL',format='%.1f'),'Sea_Days':st.column_config.NumberColumn('SEA DAYS',format='%.1f'),'Port_Days':st.column_config.NumberColumn('PORT DAYS',format='%.1f'),'Sea_Fuel':st.column_config.NumberColumn('SEA FUEL',format='%.1f'),'Port_Fuel':st.column_config.NumberColumn('PORT FUEL',format='%.1f'),'Bunkers':st.column_config.NumberColumn('BUNKERS REC',format='%.1f'),'Dist':st.column_config.NumberColumn('DIST NM',format='%d'),'Avg_Sea_Spd':st.column_config.NumberColumn('AVG SPEED',format='%.1f'),'Avg_Sea_MT_Day':st.column_config.NumberColumn('SEA MT/DAY',format='%.1f')}
                st.dataframe(vg, column_config=v_cfg, hide_index=True, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                if df['MELO_L'].sum()+df['CYLO_L'].sum()+df['GELO_L'].sum()>0:
                    st.plotly_chart(chart_lube(df),use_container_width=True,config={'displayModeBar':False})
                else: st.info('No lubricant consumption data detected.')
            with c2:
                cfig=chart_cum_drift(cum_drift)
                if cfig: st.plotly_chart(cfig,use_container_width=True,config={'displayModeBar':False})

        with tab4:
            st.plotly_chart(chart_fuel(df),use_container_width=True,config={'displayModeBar':False})
            sea_df = df[(df['Phase'] == 'SEA') & (df['Status'] == 'VERIFIED')]
            if 'AI_Exp' in sea_df.columns and sea_df['AI_Exp'].abs().sum() > 0:
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(x=sea_df['Timeline'], y=sea_df['Stoch_Var'], name='Stochastic Variance (±MT)', marker_color='rgba(123,104,238,0.2)', marker_line_color='#7b68ee', marker_line_width=1.3))
                fig_c.update_layout(**_BL, title='XGBoost Stochastic Variance Envelope (Sea Legs Only)', height=300, yaxis=dict(title='± MT', **_AX), xaxis=dict(tickangle=-45, automargin=True, **_AX))
                st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar':False})

        with tab5:
            sea_df = df[(df['Phase'] == 'SEA') & (df['Status'] == 'VERIFIED')]
            shap_ok = 'SHAP_Base' in df.columns and sea_df['SHAP_Base'].abs().sum() > 0
            
            if not shap_ok:
                st.warning("⚠️ **AI EXPLAINABILITY OFFLINE:** The AI physically isolates Port operations and requires valid Sea Legs to train.")
            else:
                anom_s = sea_df[sea_df['Status'] == 'STAT OUTLIER']
                if anom_s.empty:
                    options = sea_df['Timeline'].tolist(); sel = st.selectbox('Select Verified Sea Passage', options, key=f'shap_{vname}')
                    tr = sea_df[sea_df['Timeline']==sel].iloc[0]
                else:
                    options = anom_s['Timeline'].tolist(); sel = st.selectbox('Select Physics Anomaly', options, key=f'shap_{vname}')
                    tr = anom_s[anom_s['Timeline']==sel].iloc[0]
                
                c1, c2 = st.columns([7, 3])
                with c1:
                    eb = tr['SHAP_Base'] + tr['SHAP_Propulsion'] + tr['SHAP_Mass'] + tr['SHAP_Kinematics'] + tr['SHAP_Season']
                    fig_w = go.Figure(go.Waterfall(
                        name="SHAP", orientation="v", measure=["absolute","relative","relative","relative","relative","total"],
                        x=["Fleet Baseline", "Propulsion", "True Mass", "Kinematics", "Season", "Expected Burn"],
                        textposition="outside", text=[f"{tr['SHAP_Base']:.1f}", f"{tr['SHAP_Propulsion']:+.1f}", f"{tr['SHAP_Mass']:+.1f}", f"{tr['SHAP_Kinematics']:+.1f}", f"{tr['SHAP_Season']:+.1f}", f"{eb:.1f}"],
                        y=[tr['SHAP_Base'], tr['SHAP_Propulsion'], tr['SHAP_Mass'], tr['SHAP_Kinematics'], tr['SHAP_Season'], 0],
                        connector={"line":{"color":"rgba(201,168,76,0.15)"}},
                        decreasing={"marker":{"color":"#00e0b0"}}, increasing={"marker":{"color":"#e63946"}}, totals={"marker":{"color":"#7b68ee"}}
                    ))
                    fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40,b=20,l=0,r=0), title=dict(text=f"SHAP Derivation: {tr['Route']}", font=dict(color='#dce8f0', size=14, family='Bricolage Grotesque')), yaxis=dict(title='MT/Day', gridcolor='rgba(201,168,76,0.04)', zerolinecolor='rgba(201,168,76,0.06)'))
                    st.plotly_chart(fig_w, use_container_width=True, config={'displayModeBar':False})
                
                with c2:
                    forces = ['Propulsion', 'Mass', 'Kinematics', 'Season']
                    vals = [abs(tr['SHAP_Propulsion']), abs(tr['SHAP_Mass']), abs(tr['SHAP_Kinematics']), abs(tr['SHAP_Season'])]
                    fig_r = go.Figure(data=go.Scatterpolar(r=vals+[vals[0]], theta=forces+[forces[0]], fill='toself', fillcolor='rgba(0,224,176,0.2)', line_color='#00e0b0', line_width=2))
                    fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=20,b=20,l=20,r=20), polar=dict(radialaxis=dict(visible=False, range=[0, max(vals)*1.2 if max(vals)>0 else 1]), angularaxis=dict(tickfont=dict(color='#6d8599', size=10), gridcolor='rgba(201,168,76,0.08)')), showlegend=False)
                    st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar':False})
                
                st.info(f"**Forensic Context:** The AI isolates the pure physics of this ocean crossing. Mathematical Expected Burn: **{eb:.1f} MT/d** vs Physically Audited Burn: **{tr['Daily_Burn']:.1f} MT/d**.")

        with tab6:
            quarantined = df[df['Status'].str.contains('QUARANTINE|GHOST|OUTLIER')]
            if quarantined.empty:
                st.success("Zero anomalies detected. All chronological chains and tank soundings are intact.")
            else:
                for _,row in quarantined.iterrows():
                    s = row['Status']; c = SC.get(s, '#e63946')
                    desc = f"Fatal Audit Exception: {row['Flags']}" if 'QUARANTINE' in s else f"Anomaly Detected: {row['Flags']}"
                    st.markdown(f'<div class="acard"><div style="display:flex;justify-content:space-between;align-items:center"><div><span style="color:{c};font-weight:700;font-size:.7rem;letter-spacing:.08em;font-family:var(--fm)">{s}</span><span style="color:var(--t3);font-size:.7rem;margin-left:10px;font-family:var(--fm)">{row["Timeline"]}</span></div><span style="color:var(--t2);font-size:.68rem;font-family:var(--fb)">{row["Route"]}</span></div><div style="color:var(--t2);font-size:.7rem;margin-top:8px;line-height:1.6;font-family:var(--fb)">{desc}</div></div>',unsafe_allow_html=True)

        st.divider()
    except Exception:
        st.error(f'System Failure on file: {f.name}')
        with st.expander('View Traceback'): st.code(traceback.format_exc())

if len(fleet_results)>1:
    st.markdown('<h2 style="color:#fff;font-family:var(--fd);margin-top:10px">Fleet Comparison Matrix</h2>',unsafe_allow_html=True)
    fleet_rows=[{'Vessel':r['name'],'Legs':r['summary']['cycles'],'Verified':f"{r['summary']['integrity']:.1f}%",'DQI':int(r['summary']['avg_dqi']),'Fuel MT':r['summary']['total_fuel'],'Sea Burn':r['summary']['avg_sea_burn'],'Anomalies':r['summary']['anomalies'],'NM':int(r['summary']['total_nm']),'Days':r['summary']['total_days']} for r in fleet_results]
    st.dataframe(pd.DataFrame(fleet_rows),hide_index=True,use_container_width=True)
    
    fig_f=go.Figure()
    for r in fleet_results:
        fig_f.add_trace(go.Bar(name=r['name'],x=['Total Fuel (MT)','Sea Burn (x10)','Anomalies (x10)', 'DQI'],y=[r['summary']['total_fuel'],r['summary']['avg_sea_burn']*10,r['summary']['anomalies']*10, r['summary']['avg_dqi']]))
    fig_f.update_layout(**_BL,title='Cross-Fleet Performance',barmode='group',yaxis=dict(**_AX),xaxis=dict(**_AX))
    st.plotly_chart(fig_f,use_container_width=True,config={'displayModeBar':False})
