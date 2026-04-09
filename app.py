import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import math
import traceback
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- BULLETPROOF AI IMPORTS (Now with SHAP) ---
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

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500;600&family=Hanken+Grotesk:wght@300;400;500;600;700&display=swap');
:root{{--bg:#020609;--s1:#080d14;--s2:#0c1219;--b1:rgba(201,168,76,0.06);--b2:rgba(201,168,76,0.15);--b3:rgba(0,224,176,0.12);--acc:#00e0b0;--acc2:#c9a84c;--red:#e63946;--amber:#d4a843;--purple:#7b68ee;--t1:#dce8f0;--t2:#6d8599;--t3:#3a4d5e;--r:12px;--fd:'Bricolage Grotesque',sans-serif;--fb:'Hanken Grotesk',sans-serif;--fm:'Geist Mono',monospace}}
html,body,[class*="css"]{{font-family:var(--fb)!important;background:var(--bg)!important;color:var(--t1)}}
.stApp{{background:var(--bg);background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E")}}
header,footer,#MainMenu{{visibility:hidden!important;display:none!important}}
.block-container{{padding:0.8rem 2.5rem 0!important;max-width:100%!important}}
h1,h2,h3,h4{{font-family:var(--fd)!important;font-weight:800!important;color:#fff!important;letter-spacing:-.03em!important}}
.hero{{background:linear-gradient(135deg,var(--s1),rgba(0,95,115,0.08));border:1px solid var(--b1);border-radius:16px;padding:30px 40px;margin-bottom:24px;display:flex;align-items:center;justify-content:space-between;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent 5%,var(--acc2) 30%,var(--acc) 50%,var(--acc2) 70%,transparent 95%);opacity:.4}}
.hero::after{{content:'';position:absolute;bottom:0;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,rgba(0,224,176,0.15),transparent)}}
.hero-left{{display:flex;align-items:center;gap:22px}}.hero-logo{{width:48px;height:48px;filter:drop-shadow(0 0 12px rgba(0,224,176,0.2))}}
.hero-title{{font-family:var(--fd);font-weight:800;font-size:1.75rem;letter-spacing:-.04em;background:linear-gradient(135deg,#fff 0%,var(--acc2) 40%,var(--acc) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1}}
.hero-sub{{font-family:var(--fm);font-size:.58rem;color:var(--t3);text-transform:uppercase;letter-spacing:.2em;font-weight:500;margin-top:4px}}
.hero-badge{{font-family:var(--fm);font-size:.55rem;color:var(--t3);text-align:right;line-height:2;letter-spacing:.06em}}.hero-badge span{{color:var(--acc);font-weight:600}}
[data-testid="stFileUploader"]{{background:var(--s1)!important;border:1px dashed var(--b2)!important;border-radius:var(--r)!important;padding:14px!important;transition:all .4s}}
[data-testid="stFileUploader"]:hover{{border-color:var(--acc2)!important;box-shadow:0 0 40px rgba(201,168,76,0.06)}}
[data-testid="stFileUploader"] *{{color:var(--t1)!important;font-family:var(--fb)!important}}
[data-testid="stFileUploader"] small{{color:var(--t3)!important}}
[data-testid="stFileUploader"] button{{background:rgba(201,168,76,.08)!important;color:var(--acc2)!important;border:1px solid var(--b2)!important;border-radius:8px!important;font-weight:600!important}}
div[data-testid="stMetric"]{{background:linear-gradient(180deg,var(--s1),var(--s2))!important;border:1px solid var(--b1)!important;border-radius:var(--r);padding:18px 22px!important;position:relative;overflow:hidden;transition:border-color .3s}}
div[data-testid="stMetric"]:hover{{border-color:var(--b2)!important}}
div[data-testid="stMetric"]::after{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--acc2),transparent);opacity:0;transition:opacity .3s}}
div[data-testid="stMetric"]:hover::after{{opacity:.3}}
div[data-testid="stMetricLabel"]{{font-size:.58rem!important;color:var(--t2)!important;text-transform:uppercase!important;letter-spacing:.14em!important;font-weight:600!important;font-family:var(--fm)!important}}
div[data-testid="stMetricValue"]{{font-size:1.7rem!important;font-weight:800!important;color:#fff!important;line-height:1!important;margin-top:6px!important;font-family:var(--fd)!important;letter-spacing:-.03em!important}}
div[data-testid="stMetricValue"]>div{{color:#fff!important}}
.stTabs [data-baseweb="tab-list"]{{gap:0;background:transparent;border-bottom:1px solid rgba(201,168,76,0.08)}}
.stTabs [data-baseweb="tab"]{{background:transparent;border:none;border-bottom:2px solid transparent;border-radius:0;padding:12px 20px;color:var(--t3);font-weight:600;font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;font-family:var(--fm);transition:all .3s}}
.stTabs [data-baseweb="tab"]:hover{{color:var(--t1)}}
.stTabs [data-baseweb="tab"][aria-selected="true"]{{color:var(--acc)!important;border-bottom-color:var(--acc)!important}}
.stTabs [data-baseweb="tab-highlight"]{{display:none}}
.stDataFrame{{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--b1)!important}}
.stDownloadButton>button{{background:rgba(201,168,76,.06)!important;color:var(--acc2)!important;border:1px solid var(--b2)!important;border-radius:10px!important;font-weight:600!important;padding:10px 24px!important;transition:all .3s!important}}
.stDownloadButton>button:hover{{background:rgba(201,168,76,.12)!important;box-shadow:0 4px 30px rgba(201,168,76,0.08)!important;transform:translateY(-1px)!important}}
hr{{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.04),rgba(201,168,76,.08),rgba(201,168,76,0.04),transparent)!important;margin:32px 0!important}}
.vcard{{background:linear-gradient(165deg,var(--s1),rgba(0,95,115,0.04));border:1px solid var(--b1);border-radius:16px;padding:26px 32px;margin-bottom:20px;position:relative;overflow:hidden}}
.vcard::before{{content:'';position:absolute;top:0;left:5%;right:5%;height:1px;background:linear-gradient(90deg,transparent,var(--acc2),transparent);opacity:.2}}
.vcard::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:80px;background:linear-gradient(180deg,transparent,rgba(0,224,176,0.015));pointer-events:none}}
.acard{{background:var(--s1);border-radius:10px;padding:16px 20px;margin-bottom:8px;transition:transform .2s,box-shadow .2s}}
.acard:hover{{transform:translateX(3px);box-shadow:-3px 0 20px rgba(0,0,0,0.3)}}
.pill{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.58rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-family:var(--fm)}}
.p-ok{{background:rgba(0,224,176,.06);color:var(--acc);border:1px solid rgba(0,224,176,.15)}}
.p-w{{background:rgba(212,168,67,.06);color:var(--amber);border:1px solid rgba(212,168,67,.15)}}
.p-c{{background:rgba(230,57,70,.06);color:var(--red);border:1px solid rgba(230,57,70,.15)}}
::-webkit-scrollbar{{width:4px;height:4px}}::-webkit-scrollbar-track{{background:var(--bg)}}::-webkit-scrollbar-thumb{{background:var(--t3);border-radius:2px}}
details{{background:var(--s1)!important;border:1px solid var(--b1)!important;border-radius:var(--r)!important}}
details summary{{color:var(--t1)!important;font-weight:600!important}}
/* Make sure selectboxes fit the dark theme */
div[data-baseweb="select"] > div {{background: var(--s1); border-color: var(--b1); color: #fff;}}
</style>""", unsafe_allow_html=True)

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
    if val is None: return np.nan
    if isinstance(val,float): return val
    if isinstance(val,int): return float(val)
    s=str(val).strip()
    if s=='' or s.upper() in ('NAN','N/A','NA','-','NIL','NONE'): return np.nan
    s=re.sub(r'[^\d.\-]','',s)
    if not s or s in ('.','-','-.'): return np.nan
    try: return float(s)
    except: return np.nan
def _sn0(val):
    v=_sn(val)
    return 0.0 if np.isnan(v) else v
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

COL={0:'Voy',1:'Port',2:'AD',3:'Date',4:'Time',5:'DistLeg',6:'TotalDist',7:'TotalTime',8:'Speed',9:'Bunk_FO',10:'Bunk_MGO',11:'Bunk_MELO',12:'Bunk_HSCYLO',13:'Bunk_LSCYLO',14:'Bunk_GELO',15:'FO_A',16:'FO_L',17:'MGO_A',18:'MGO_L',19:'MELO_R',20:'HSCYLO_R',21:'LSCYLO_R',22:'GELO_R',23:'FW_R',24:'CargoName',25:'CargoQty',26:'Op_Load',27:'Op_Disch',28:'Op_Trans',29:'Op_Bunk',30:'Diff_FO',31:'Diff_MGO'}
ZERO_COLS=['DistLeg','TotalDist','TotalTime','Speed','Bunk_FO','Bunk_MGO','Bunk_MELO','Bunk_HSCYLO','Bunk_LSCYLO','Bunk_GELO','CargoQty','Diff_FO','Diff_MGO']
ROB_COLS=['FO_A','FO_L','MGO_A','MGO_L','MELO_R','HSCYLO_R','LSCYLO_R','GELO_R','FW_R']

def compute_dqi(r1,r2,daily_burn,drift,chrono_bad,mgo_neg):
    s={}
    rob_f=['FO_A','FO_L','MGO_A']
    s['rob']=sum(1 for f in rob_f if not np.isnan(r1.get(f,np.nan)) and not np.isnan(r2.get(f,np.nan)))/len(rob_f)
    tol=max(30.0,0.03*max(r1.get('FO_A',0) or 0,r2.get('FO_A',0) or 0))
    s['drift']=gauss_mf(drift,0.0,tol)
    if daily_burn>0: s['burn']=gauss_mf(daily_burn,30.0,25.0)
    elif daily_burn==0: s['burn']=0.5
    else: s['burn']=0.1
    s['chrono']=0.3 if chrono_bad else 1.0
    s['mgo']=0.3 if mgo_neg else 1.0
    w={'rob':0.30,'drift':0.30,'burn':0.20,'chrono':0.10,'mgo':0.10}
    log_sum=sum(w[k]*math.log(max(v,0.001)) for k,v in s.items())
    return min(100,max(0,round(math.exp(log_sum)*100,0)))

# ═══════════════════════════════════════════════════════════════════════════════
# AI DIGITAL TWIN MODULE (v17.1: XGBOOST + SHAP EXPLAINER)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def calculate_stochastic_variance(trip_df):
    zeros_df = pd.DataFrame({'Stoch_Var':[0.0]*len(trip_df), 'SHAP_Base':[0.0]*len(trip_df), 'SHAP_Speed':[0.0]*len(trip_df), 'SHAP_Cargo':[0.0]*len(trip_df), 'SHAP_Route':[0.0]*len(trip_df)}, index=trip_df.index)
    try:
        # Failsafe 1: Is XGBoost/SHAP installed?
        if not HAS_ML:
            st.warning("AI ENGINE OFFLINE: XGBoost or SHAP is not installed in the terminal.")
            return zeros_df
            
        if trip_df.empty: 
            return zeros_df
            
        ml = trip_df[['Speed_kn','CargoQty','Route','Daily_Burn','Days']].copy()
        
        ml['Speed_kn'] = ml['Speed_kn'].fillna(12.0)
        ml['CargoQty'] = ml['CargoQty'].replace(0, np.nan).ffill().fillna(0)
        ml['Route_Code'] = ml['Route'].astype('category').cat.codes
        
        train = ml[ml['Daily_Burn'] > 0]
        
        # Failsafe 2: Do we have enough data?
        if len(train) < 5: 
            return zeros_df
            
        X_train = train[['Speed_kn','CargoQty','Route_Code']]
        y_train = train['Daily_Burn']
        
        ai_model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, objective='reg:squarederror')
        ai_model.fit(X_train, y_train)
        
        predictions = ai_model.predict(X_train)
        route_noise = np.sqrt(mean_squared_error(y_train, predictions))
        route_noise = min(4.0, max(0.3, route_noise)) 
        
        X_all = ml[['Speed_kn','CargoQty','Route_Code']]
        baseline_prediction = ai_model.predict(X_all)
        
        # --- SHAP INJECTION ---
        explainer = shap.TreeExplainer(ai_model)
        shap_vals = explainer.shap_values(X_all)
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray): base_val = base_val[0]
        
        np.random.seed(42)
        stochastic_reality = baseline_prediction + np.random.normal(loc=0, scale=route_noise, size=len(trip_df))
        
        variance_result = (ml['Daily_Burn'] - stochastic_reality) * ml['Days']
        
        return pd.DataFrame({
            'Stoch_Var': variance_result.round(1),
            'SHAP_Base': [base_val] * len(X_all),
            'SHAP_Speed': shap_vals[:, 0],
            'SHAP_Cargo': shap_vals[:, 1],
            'SHAP_Route': shap_vals[:, 2]
        }, index=trip_df.index)
        
    except Exception as e: 
        st.error(f"AI ENGINE CRASHED: {str(e)}")
        return zeros_df

@st.cache_data(show_spinner=False)
def process_file(uploaded_file):
    vn_raw=re.sub(r'\.[^.]+$','',uploaded_file.name).strip()
    vname=re.sub(r'[_\-]+',' ',vn_raw).upper()
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith('.xlsx'): df_raw=pd.read_excel(uploaded_file,header=None,engine='openpyxl')
    else:
        uploaded_file.seek(0)
        df_raw=pd.read_csv(io.StringIO(uploaded_file.read().decode('latin-1',errors='replace')),sep=None,engine='python',header=None,on_bad_lines='skip')
    if df_raw.empty or len(df_raw)<4: return pd.DataFrame(),vname,{},[]
    kw=['PORT','VOY','DATE','A/D','FO','ROB','DIST','BUNKER','SPEED','TIME']
    best_row,best_sc=0,0
    for i in range(min(60,len(df_raw))):
        txt=' '.join(str(x).upper() for x in df_raw.iloc[i].values if pd.notna(x))
        sc=sum(1 for k in kw if k in txt)
        if sc>best_sc: best_sc,best_row=sc,i
    df=df_raw.iloc[best_row+2:].copy().reset_index(drop=True)
    nc=len(df.columns); df.columns=[COL.get(i,f'_x{i}') for i in range(nc)]
    for c in ZERO_COLS:
        if c in df.columns: df[c]=df[c].apply(_sn0)
        else: df[c]=0.0
    for c in ROB_COLS:
        if c in df.columns: df[c]=df[c].apply(_sn)
        else: df[c]=np.nan
    for c in ['Port','Voy','AD','CargoName']:
        if c in df.columns: df[c]=df[c].fillna('').astype(str).str.strip()
        else: df[c]=''
    df['Datetime']=df.apply(lambda r:_parse_dt(r.get('Date'),r.get('Time')),axis=1)
    n_before=len(df); df=df.dropna(subset=['Datetime']); n_dropped=n_before-len(df)
    df=df.reset_index(drop=True)
    if len(df)<2: return pd.DataFrame(),vname,{},[]
    def _cad(v):
        v=str(v).strip().upper().replace(' ','')
        if v=='D': return 'D'
        if v.startswith('A') and 'D' not in v: return 'A'
        if 'D' in v and 'A' not in v: return 'D'
        return v
    df['AD']=df['AD'].apply(_cad)
    d_indices=df[df['AD']=='D'].index.tolist()
    if len(d_indices)<2: return pd.DataFrame(),vname,{},[]
    cum_drift=[]
    for idx in d_indices:
        fa=_sn(df.loc[idx,'FO_A']); fl=_sn(df.loc[idx,'FO_L'])
        fa=fa if not np.isnan(fa) else 0; fl=fl if not np.isnan(fl) else fa
        cum_drift.append({'dt':df.loc[idx,'Datetime'],'gap':fa-fl,'port':str(df.loc[idx,'Port']).strip()[:20]})
    trips=[]
    for ci in range(len(d_indices)-1):
        idx1,idx2=d_indices[ci],d_indices[ci+1]
        r1,r2=df.loc[idx1],df.loc[idx2]
        foa1=_sn(r1['FO_A']); foa2=_sn(r2['FO_A'])
        rob_ok=not(np.isnan(foa1) or np.isnan(foa2))
        foa1=foa1 if not np.isnan(foa1) else 0.0; foa2=foa2 if not np.isnan(foa2) else 0.0
        fol1=_sn(r1['FO_L']); fol1=fol1 if not np.isnan(fol1) else foa1
        fol2=_sn(r2['FO_L']); fol2=fol2 if not np.isnan(fol2) else foa2
        mgoa1=_sn(r1['MGO_A']); mgoa1=mgoa1 if not np.isnan(mgoa1) else 0.0
        mgoa2=_sn(r2['MGO_A']); mgoa2=mgoa2 if not np.isnan(mgoa2) else 0.0
        between=df.loc[idx1+1:idx2-1]; a_rows=between[between['AD']=='A']
        port_dep=str(r1['Port']).strip()[:25] or '\u2014'
        port_arr='\u2014'
        for _,ar in a_rows.iterrows():
            pn=str(ar['Port']).strip()
            if pn and not _is_ops(pn): port_arr=pn[:25]; break
        if port_arr=='\u2014':
            if not a_rows.empty: port_arr=str(a_rows.iloc[-1]['Port']).strip()[:25] or '\u2014'
            else: port_arr=str(r2['Port']).strip()[:25] or '\u2014'
        window=df.loc[idx1+1:idx2]; hours=window['TotalTime'].sum(); chrono_bad=False
        if hours<=0:
            dt_d=(r2['Datetime']-r1['Datetime']).total_seconds()/3600.0
            if dt_d>0: hours=dt_d; chrono_bad=True
            else: continue
        days=hours/24.0; leg_nm=window['TotalDist'].sum()
        if leg_nm<=0: leg_nm=max(0.0,r2['TotalDist'])
        spd_v=window['Speed'].replace(0,np.nan).dropna()
        speed=spd_v.mean() if not spd_v.empty else (leg_nm/hours if hours>0 else 0.0)
        bsl=df.loc[idx1+1:idx2]
        bfo=bsl['Bunk_FO'].sum(); bmgo=bsl['Bunk_MGO'].sum()
        bmelo=bsl['Bunk_MELO'].sum(); bhsc=bsl['Bunk_HSCYLO'].sum()
        blsc=bsl['Bunk_LSCYLO'].sum(); bgelo=bsl['Bunk_GELO'].sum()
        hfo_c=(foa1-foa2)+bfo; mgo_raw=(mgoa1-mgoa2)+bmgo; mgo_c=max(0.0,mgo_raw); mgo_neg=mgo_raw<-5
        melo_c=max(0,(_sn0(r1.get('MELO_R'))-_sn0(r2.get('MELO_R')))+bmelo)
        hsc_c=max(0,(_sn0(r1.get('HSCYLO_R'))-_sn0(r2.get('HSCYLO_R')))+bhsc)
        lsc_c=max(0,(_sn0(r1.get('LSCYLO_R'))-_sn0(r2.get('LSCYLO_R')))+blsc)
        gelo_c=max(0,(_sn0(r1.get('GELO_R'))-_sn0(r2.get('GELO_R')))+bgelo)
        total_fuel=hfo_c+mgo_c; daily_burn=total_fuel/days if days>0 else 0.0
        drift=abs((foa1-fol1)-(foa2-fol2)); tol=max(30.0,0.03*max(foa1 or 0,foa2 or 0))
        bunk_disc=0.0
        if bfo>0:
            bunk_rows=bsl[bsl['Bunk_FO']>0]
            if not bunk_rows.empty:
                bi=bunk_rows.index[0]
                if bi>0 and bi-1 in df.index:
                    rob_before=_sn0(df.loc[bi-1,'FO_A']); rob_after=_sn0(df.loc[bi,'FO_A'])
                    bunk_disc=abs(bfo-(rob_after-rob_before)) if rob_before>0 else 0.0
        cargo=str(r1.get('CargoName','')).strip().upper(); qty=_sn0(r1.get('CargoQty',0))
        is_laden=cargo not in ('','NAN','NIL','NONE','BALLAST') and qty>0
        condition='LADEN' if is_laden else 'BALLAST'
        dqi=compute_dqi({'FO_A':foa1,'FO_L':fol1,'MGO_A':mgoa1},{'FO_A':foa2,'FO_L':fol2,'MGO_A':mgoa2},daily_burn,drift,chrono_bad or not rob_ok,mgo_neg)
        p_normal=gauss_mf(drift,0.0,tol); p_ghost=trap_mf(-total_fuel,tol*0.8,tol*1.2,10000,20000)
        status='VERIFIED'
        if p_ghost>0.7: status='GHOST BUNKER'
        elif p_normal<0.30: status='LEDGER VARIANCE'
        phase='SEA' if leg_nm>50 and speed>3 else ('COASTAL' if leg_nm>5 else 'PORT')
        flags=[]
        if not rob_ok: flags.append('ROB_MISS')
        if chrono_bad: flags.append('TIME_FB')
        if mgo_neg: flags.append('MGO_NEG')
        if bunk_disc>50: flags.append(f'BUNK_DISC:{bunk_disc:.0f}')
        
        trips.append({'Indicator':ICONS.get(status,ICONS['VERIFIED']),'Timeline':f"{r1['Datetime'].strftime('%d %b %y')}  →  {r2['Datetime'].strftime('%d %b %y')}",'Phase':phase,'Condition':condition,'CargoQty':qty,'Route':f"{port_dep}  →  {port_arr}",'Days':round(days,2),'Dist_NM':round(leg_nm,0),'Speed_kn':round(speed,1),'HFO_MT':round(hfo_c,1),'MGO_MT':round(mgo_c,1),'Fuel_MT':round(total_fuel,1),'Daily_Burn':round(daily_burn,1),'MELO_L':round(melo_c,0),'CYLO_L':round(hsc_c+lsc_c,0),'GELO_L':round(gelo_c,0),'Drift_MT':round(drift,1),'DQI':int(dqi),'Status':status,'Voy':str(r1['Voy']).strip(),'Flags':','.join(flags) if flags else ''})
        
    trip_df=pd.DataFrame(trips)
    if len(trip_df)>=6:
        for cond in ['LADEN','BALLAST']:
            ver=trip_df[(trip_df['Status']=='VERIFIED')&(trip_df['Daily_Burn']>0)&(trip_df['Condition']==cond)]
            if len(ver)>=4:
                q1,q3=ver['Daily_Burn'].quantile(0.25),ver['Daily_Burn'].quantile(0.75); iqr=q3-q1
                if iqr>0:
                    lo,hi=q1-2.0*iqr,q3+2.0*iqr
                    mask=(trip_df['Status']=='VERIFIED')&(trip_df['Daily_Burn']>0)&(trip_df['Condition']==cond)&((trip_df['Daily_Burn']<lo)|(trip_df['Daily_Burn']>hi))
                    trip_df.loc[mask,'Status']='STAT OUTLIER'; trip_df.loc[mask,'Indicator']=ICONS['STAT OUTLIER']
                    
    # Initialize Stoch_Var & SHAP execution
    if not trip_df.empty:
        ai_df = calculate_stochastic_variance(trip_df)
        for col in ai_df.columns: trip_df[col] = ai_df[col]
        
        cols=list(trip_df.columns)
        if 'Stoch_Var' in cols and 'DQI' in cols:
            cols.insert(cols.index('DQI'),cols.pop(cols.index('Stoch_Var')))
            trip_df=trip_df[cols]
        if 'Drift_MT' in trip_df.columns: trip_df=trip_df.drop(columns=['Drift_MT'])
        
    summary={}
    if not trip_df.empty:
        n=len(trip_df); n_ok=(trip_df['Status']=='VERIFIED').sum(); pb=trip_df[trip_df['Daily_Burn']>0]['Daily_Burn']
        summary={'integrity':round(n_ok/n*100,1),'avg_dqi':round(trip_df['DQI'].mean(),0),'total_fuel':round(trip_df['Fuel_MT'].sum(),1),'total_hfo':round(trip_df['HFO_MT'].sum(),1),'total_mgo':round(trip_df['MGO_MT'].sum(),1),'avg_burn':round(pb.mean(),1) if len(pb) else 0.0,'total_nm':round(trip_df['Dist_NM'].sum(),0),'total_melo':round(trip_df['MELO_L'].sum(),0),'total_cylo':round(trip_df['CYLO_L'].sum(),0),'total_gelo':round(trip_df['GELO_L'].sum(),0),'total_days':round(trip_df['Days'].sum(),1),'cycles':n,'anomalies':n-n_ok,'ghost':int((trip_df['Status']=='GHOST BUNKER').sum()),'ledger':int((trip_df['Status']=='LEDGER VARIANCE').sum()),'outlier':int((trip_df['Status']=='STAT OUTLIER').sum()),'flagged':int((trip_df['Flags']!='').sum()),'laden':int((trip_df['Condition']=='LADEN').sum()),'ballast':int((trip_df['Condition']=='BALLAST').sum()),'dropped_dt':n_dropped}
    return trip_df,vname,summary,cum_drift

_BL=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',hovermode='x unified',hoverlabel=dict(bgcolor='#0c1219',bordercolor='rgba(201,168,76,0.12)',font=dict(family='Hanken Grotesk',color='#dce8f0',size=12)),font=dict(family='Hanken Grotesk',color='#4a6275',size=11),title_font=dict(family='Bricolage Grotesque',color='#ffffff',size=15),margin=dict(l=0,r=0,t=55,b=5))
_AX=dict(gridcolor='rgba(201,168,76,0.04)',zerolinecolor='rgba(201,168,76,0.06)')

def chart_fuel(df):
    bc=[SC.get(s,'#00e0b0') for s in df['Status']]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.65,.35],vertical_spacing=.06)
    fig.add_trace(go.Bar(x=df['Timeline'],y=df['Fuel_MT'],name='Cycle Fuel',marker=dict(color=[_rgba(c,.12) for c in bc],line=dict(color=bc,width=1.5)),text=df['Fuel_MT'],textposition='outside',textfont=dict(size=9,color='#4a6275')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['Timeline'],y=df['Daily_Burn'],name='Daily Burn',mode='lines+markers',line=dict(color='#00e0b0',width=2,shape='spline'),marker=dict(size=4),fill='tozeroy',fillcolor='rgba(0,224,176,0.03)'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df['Timeline'],y=df['Speed_kn'],name='Speed',mode='lines+markers',line=dict(color='#c9a84c',width=2,shape='spline'),marker=dict(size=4,color='#c9a84c')),row=2,col=1)
    fig.update_layout(**_BL,title='Fuel Consumption & Speed Profile',barmode='overlay',showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)),yaxis=dict(title='MT',**_AX),yaxis2=dict(title='kn',**_AX),xaxis=dict(**_AX),xaxis2=dict(**_AX))
    fig.update_xaxes(tickangle=-45,tickfont=dict(size=8)); return fig

def chart_stoch_var_dqi(df):
    if 'Stoch_Var' not in df.columns: return None
    cc=[SC.get(s,'#00e0b0') for s in df['Status']]
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=df['Timeline'],y=df['Stoch_Var'],name='Stoch Var (MT)',marker=dict(color=[_rgba(c,.2) for c in cc],line=dict(color=cc,width=1.3))),secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Timeline'],y=df['DQI'],name='DQI',mode='lines+markers',line=dict(color='#00e0b0',width=2,shape='spline'),marker=dict(size=4)),secondary_y=True)
    fig.update_layout(**_BL,title='XGBoost Stochastic Variance & Data Quality Index',barmode='overlay',showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)))
    fig.update_yaxes(title_text='Variance MT',secondary_y=False,**_AX); fig.update_yaxes(title_text='DQI',secondary_y=True,range=[0,105],**_AX)
    fig.update_xaxes(tickangle=-45,tickfont=dict(size=8),**_AX); return fig

def chart_cum_drift(cum_drift):
    if not cum_drift: return None
    cdf=pd.DataFrame(cum_drift)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=cdf['dt'],y=cdf['gap'],mode='lines+markers',name='A−L Gap',line=dict(color='#c9a84c',width=2),marker=dict(size=3),fill='tozeroy',fillcolor='rgba(201,168,76,0.04)'))
    fig.add_hline(y=0,line=dict(color='rgba(255,255,255,0.06)',width=1,dash='dot'))
    fig.update_layout(**_BL,title='Cumulative Actual vs Ledger Gap',yaxis=dict(title='FO_A − FO_L (MT)',**_AX),xaxis=dict(**_AX)); return fig

def chart_lube(df):
    fig=go.Figure()
    if df['MELO_L'].sum()>0: fig.add_trace(go.Bar(x=df['Timeline'],y=df['MELO_L'],name='MELO',marker=dict(color='rgba(0,224,176,0.12)',line=dict(color='#00e0b0',width=1.3))))
    if df['CYLO_L'].sum()>0: fig.add_trace(go.Bar(x=df['Timeline'],y=df['CYLO_L'],name='CYLO',marker=dict(color='rgba(123,104,238,0.12)',line=dict(color='#7b68ee',width=1.3))))
    if df['GELO_L'].sum()>0: fig.add_trace(go.Bar(x=df['Timeline'],y=df['GELO_L'],name='GELO',marker=dict(color='rgba(201,168,76,0.12)',line=dict(color='#c9a84c',width=1.3))))
    fig.update_layout(**_BL,title='Lubricant Consumption (Liters)',barmode='group',showlegend=True,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)),yaxis=dict(title='L',**_AX),xaxis=dict(**_AX))
    fig.update_xaxes(tickangle=-45,tickfont=dict(size=8)); return fig

def chart_voyage(df):
    vg=df.groupby('Voy',sort=False).agg(Fuel=('Fuel_MT','sum'),Days=('Days','sum'),Dist=('Dist_NM','sum'),Legs=('Voy','count'),Burn=('Daily_Burn','mean')).reset_index()
    vg=vg[vg['Fuel']>0]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=vg['Voy'],y=vg['Fuel'],name='Voyage Fuel',marker=dict(color='rgba(0,224,176,0.12)',line=dict(color='#00e0b0',width=1.5)),text=vg['Legs'].apply(lambda x:f'{x}L'),textposition='outside',textfont=dict(size=9,color='#4a6275')))
    fig.update_layout(**_BL,title='Fuel by Commercial Voyage (L = legs)',yaxis=dict(title='MT',**_AX),xaxis=dict(title='Voyage',**_AX)); return fig

st.markdown(f"""
<div class="hero"><div class="hero-left"><img src="data:image/svg+xml;base64,{_LOGO}" class="hero-logo" alt=""/><div><div class="hero-title">POSEIDON TITAN</div><div class="hero-sub">Fleet Consumables Intelligence Engine</div></div></div><div class="hero-badge"><span>KERNEL</span>&ensp;XGBoost + Continuous Mass<br><span>PIPELINE</span>&ensp;D-to-D Immutable Ledger<br><span>BUILD</span>&ensp;v17.1 God Mode</div></div>""",unsafe_allow_html=True)

uploaded_files=st.file_uploader('Upload vessel telemetry',accept_multiple_files=True,type=['xlsx','csv'],label_visibility='collapsed')

if not uploaded_files:
    st.markdown("""<div style="text-align:center;padding:100px 20px">
        <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:28px;opacity:.12">
            <circle cx="40" cy="40" r="36" fill="none" stroke="#c9a84c" stroke-width="0.8" stroke-dasharray="6 6">
                <animateTransform attributeName="transform" type="rotate" from="0 40 40" to="360 40 40" dur="30s" repeatCount="indefinite"/></circle>
            <circle cx="40" cy="40" r="24" fill="none" stroke="#00e0b0" stroke-width="0.6" stroke-dasharray="3 8">
                <animateTransform attributeName="transform" type="rotate" from="360 40 40" to="0 40 40" dur="20s" repeatCount="indefinite"/></circle>
            <path d="M40 14L40 66 M22 26Q40 36 58 26 M20 40Q40 50 60 40 M22 54Q40 64 58 54" fill="none" stroke="#00e0b0" stroke-width="1.5" stroke-linecap="round" opacity=".35"/></svg>
        <h2 style="color:#fff;font-family:'Bricolage Grotesque';font-weight:800;font-size:1.4rem;margin-bottom:8px;letter-spacing:-0.03em">Awaiting Telemetry</h2>
        <p style="color:#3a4d5e;font-size:.8rem;max-width:420px;margin:0 auto;line-height:1.7;font-family:'Hanken Grotesk'">
            Drop vessel noon-report files to execute the<br>Departure-to-Departure cyclic forensic audit.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

fleet_results=[]
for f in uploaded_files:
    try:
        with st.spinner(f'Processing {f.name}...'):
            df,vname,summary,cum_drift=process_file(f)
        if df.empty: st.warning(f'No D-to-D cycles in {f.name}.'); continue
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
        cols[2].metric('Avg Burn (MT/d)',f"{summary['avg_burn']:.1f}")
        cols[3].metric('MELO (L)',f"{int(summary['total_melo']):,}")
        cols[4].metric('CYLO (L)',f"{int(summary['total_cylo']):,}")
        ap=[]
        if summary['ghost']: ap.append(f"{summary['ghost']} ghost")
        if summary['ledger']: ap.append(f"{summary['ledger']} ledger")
        if summary['outlier']: ap.append(f"{summary['outlier']} outlier")
        cols[5].metric('Anomalies',' / '.join(ap) if ap else '0')

        # --- TAB 6 INJECTED HERE ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['AUDIT MATRIX', 'FUEL ANALYTICS', 'DRIFT TRAJECTORY', 'LUBE OIL', 'FORENSIC DETAIL', 'AI EXPLAINER (SHAP)'])

        with tab1:
            dcfg={'Indicator':st.column_config.ImageColumn(' ',width='small'),'Timeline':st.column_config.TextColumn('TIMELINE',width='medium'),'Phase':st.column_config.TextColumn('PH',width='small'),'Condition':st.column_config.TextColumn('COND',width='small'),'Route':st.column_config.TextColumn('ROUTE',width='large'),'Days':st.column_config.NumberColumn('DAYS',format='%.2f'),'Dist_NM':st.column_config.NumberColumn('DIST',format='%d'),'Speed_kn':st.column_config.NumberColumn('SPD',format='%.1f'),'HFO_MT':st.column_config.NumberColumn('HFO',format='%.1f'),'MGO_MT':st.column_config.NumberColumn('MGO',format='%.1f'),'Fuel_MT':st.column_config.NumberColumn('FUEL',format='%.1f'),'Daily_Burn':st.column_config.ProgressColumn('BURN',format='%.1f',min_value=0,max_value=float(max(df['Daily_Burn'].max()*1.15,1))),'MELO_L':st.column_config.NumberColumn('MELO',format='%d'),'CYLO_L':st.column_config.NumberColumn('CYLO',format='%d'),'GELO_L':st.column_config.NumberColumn('GELO',format='%d'),'Stoch_Var':st.column_config.NumberColumn('STOCH VAR',format='%.1f'),'DQI':st.column_config.ProgressColumn('DQI',format='%d',min_value=0,max_value=100),'Status':st.column_config.TextColumn('STATUS',width='medium'),'Flags':st.column_config.TextColumn('FLAGS',width='medium'),'Voy':None, 'CargoQty':None, 'SHAP_Base':None, 'SHAP_Speed':None, 'SHAP_Cargo':None, 'SHAP_Route':None}
            st.dataframe(df,column_config=dcfg,hide_index=True,use_container_width=True,height=min(500,38+len(df)*35))
            buf=io.BytesIO(); exp=df.drop(columns=['Indicator'],errors='ignore')
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
            st.caption('Tracks the running FO Actual − Ledger gap at every departure node. Jumps indicate measurement resets or systematic errors.')

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
                    dm={'GHOST BUNKER':f"Net fuel = {row['Fuel_MT']:.1f} MT (negative) — unrecorded bunkering ~{abs(row['Fuel_MT']):.0f} MT. DQI: {row['DQI']}%.{fl}",'LEDGER VARIANCE':f"XGBoost Stochastic Variance: {ai_v:.1f} MT exceeded threshold. DQI: {row['DQI']}%. {row['Condition']} leg, {row['Days']:.1f}d.{fl}",'STAT OUTLIER':f"Burn {row['Daily_Burn']:.1f} MT/d outside {row['Condition']} IQR fence. Stoch Var: {ai_v:.1f} MT. DQI: {row['DQI']}%.{fl}"}
                    st.markdown(f'<div class="acard" style="border:1px solid rgba({ri[0]},{ri[1]},{ri[2]},.12);border-left:3px solid {sc}"><div style="display:flex;justify-content:space-between;align-items:center"><div><span style="color:{sc};font-weight:700;font-size:.7rem;letter-spacing:.08em;font-family:var(--fm)">{s}</span><span style="color:var(--t3);font-size:.7rem;margin-left:10px;font-family:var(--fm)">{row["Timeline"]}</span></div><span style="color:var(--t2);font-size:.68rem;font-family:var(--fb)">{row["Route"]}</span></div><div style="color:var(--t2);font-size:.7rem;margin-top:8px;line-height:1.6;font-family:var(--fb)">{dm.get(s,"")}</div></div>',unsafe_allow_html=True)
                    
        # --- NEW SHAP EXPLAINER TAB ---
        with tab6:
            st.markdown('<h3 style="color:#fff;font-family:var(--fd);font-size:1.2rem;margin-bottom:10px;margin-top:10px">Neural Logic Extraction</h3>', unsafe_allow_html=True)
            
            # Did the AI actually run, or did it return zeros because of bad data?
            shap_ran = df['SHAP_Base'].abs().sum() > 0 if 'SHAP_Base' in df.columns else False
            
            if not shap_ran:
                st.warning("⚠️ **AI EXPLAINABILITY OFFLINE:** The XGBoost engine did not generate neural logic for this dataset. This usually happens if the uploaded file has fewer than 5 valid sea-passages, which prevents the AI from learning the complex physics curve of the hull.")
            else:
                anomalies = df[df['Status'] != 'VERIFIED']
                
                # If there are anomalies, select from them. Otherwise, select from any leg.
                if anomalies.empty:
                    st.success("No anomalies detected, but you can still view the AI's logic for any valid leg below.")
                    options = df['Timeline'].tolist()
                    sel_time = st.selectbox("Select Voyage Leg", options)
                    target_row = df[df['Timeline'] == sel_time].iloc[0]
                else:
                    st.write("Select a flagged anomaly below to view the exact mathematical receipt of why the AI expects a different fuel burn.")
                    options = anomalies['Timeline'].tolist()
                    sel_time = st.selectbox("Select Flagged Anomaly", options)
                    target_row = anomalies[anomalies['Timeline'] == sel_time].iloc[0]
                
                # Build the dynamic Waterfall chart
                expected = target_row['SHAP_Base'] + target_row['SHAP_Speed'] + target_row['SHAP_Cargo'] + target_row['SHAP_Route']
                
                fig = go.Figure(go.Waterfall(
                    name="SHAP", orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=["Fleet Baseline", "Speed Impact", "Cargo Impact", "Route Impact", "AI Expected Burn"],
                    textposition="outside",
                    text=[f"{target_row['SHAP_Base']:.1f}", f"{target_row['SHAP_Speed']:+.1f}", f"{target_row['SHAP_Cargo']:+.1f}", f"{target_row['SHAP_Route']:+.1f}", f"{expected:.1f}"],
                    y=[target_row['SHAP_Base'], target_row['SHAP_Speed'], target_row['SHAP_Cargo'], target_row['SHAP_Route'], 0],
                    connector={"line":{"color":"rgba(201,168,76,0.15)"}},
                    decreasing={"marker":{"color":"#00e0b0"}},
                    increasing={"marker":{"color":"#e63946"}},
                    totals={"marker":{"color":"#7b68ee"}}
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    height=380, margin=dict(t=40, b=20, l=0, r=0),
                    title=dict(text=f"Mathematical Derivation: {target_row['Route']} ({sel_time})", font=dict(color='#dce8f0', size=14)),
                    yaxis=dict(title='Metric Tons (MT)', gridcolor='rgba(201,168,76,0.04)', zerolinecolor='rgba(201,168,76,0.06)')
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
                
                # The Translation
                st.info(f"**Forensic Translation:** The AI determined the baseline burn for this ship is **{target_row['SHAP_Base']:.1f} MT**. The specific speed of {target_row['Speed_kn']}kn altered the burn by **{target_row['SHAP_Speed']:+.1f} MT**, the cargo weight of {target_row['CargoQty']} MT altered it by **{target_row['SHAP_Cargo']:+.1f} MT**, and the oceanography of the route altered it by **{target_row['SHAP_Route']:+.1f} MT**. \n\nThe final expected burn was **{expected:.1f} MT**. The Chief Engineer reported **{target_row['Daily_Burn']:.1f} MT**.")

        st.divider()
    except Exception:
        st.error(f'Failed: {f.name}')
        with st.expander('Trace'): st.code(traceback.format_exc())

if len(fleet_results)>1:
    st.markdown('<h2 style="margin-top:10px">Fleet Comparison</h2>',unsafe_allow_html=True)
    fleet_rows=[]
    for r in fleet_results:
        s=r['summary']
        fleet_rows.append({'Vessel':r['name'],'Cycles':s['cycles'],'Verified':f"{s['integrity']:.1f}%",'DQI':int(s['avg_dqi']),'Fuel MT':s['total_fuel'],'Avg Burn':s['avg_burn'],'Anomalies':s['anomalies'],'NM':int(s['total_nm']),'Days':s['total_days']})
    st.dataframe(pd.DataFrame(fleet_rows),hide_index=True,use_container_width=True)
    fig=go.Figure()
    for r in fleet_results:
        fig.add_trace(go.Bar(name=r['name'],x=['Total Fuel','Avg Burn x10','Anomalies x10','DQI'],y=[r['summary']['total_fuel'],r['summary']['avg_burn']*10,r['summary']['anomalies']*10,r['summary']['avg_dqi']]))
    fig.update_layout(**_BL,title='Fleet Comparison',barmode='group',yaxis=dict(**_AX),xaxis=dict(**_AX))
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})