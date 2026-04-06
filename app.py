import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & MASTER CSS (Glass & Grain)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="POSEIDON • Command", page_icon="⚓", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root { --ocean-dark: #020813; --teal-glow: #00d4aa; --gold: #f5c842; --danger: #ff4560; --success: #00e396; --text-primary: #e8f4fd; --text-muted: #6b8cae; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; max-width: 96% !important; }

.stApp {
  background-color: var(--ocean-dark);
  background-image: 
    radial-gradient(at 0% 0%, rgba(0, 212, 170, 0.08) 0px, transparent 50%),
    radial-gradient(at 100% 0%, rgba(0, 143, 251, 0.08) 0px, transparent 50%),
    url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.05'/%3E%3C/svg%3E");
  color: var(--text-primary);
}

header, footer { visibility: hidden !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero-banner { background: rgba(10,25,47,0.4); backdrop-filter: blur(20px); border: 1px solid rgba(0,212,170,0.2); border-radius: 16px; padding: 30px 40px; margin-bottom: 24px; box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3); }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #e8f4fd 0%, var(--teal-glow) 50%, var(--gold) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 8px 0; }

div[data-testid="metric-container"], .kpi-card { background: rgba(10, 25, 47, 0.4) !important; backdrop-filter: blur(20px) saturate(180%); border: 1px solid rgba(255, 255, 255, 0.05) !important; border-radius: 14px; padding: 20px; box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05), 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important; }
.kpi-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--text-primary); line-height: 1; }
.kpi-delta { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--text-muted); margin-top: 6px; }

.section-header { display: flex; align-items: center; gap: 16px; margin: 30px 0 16px 0; }
.section-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,212,170,0.3), transparent); }
.section-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: var(--teal-glow); letter-spacing: 0.1em; }

.stDataFrame { border-radius: 10px !important; overflow: hidden !important; border: 1px solid rgba(0,212,170,0.1) !important; }
.streamlit-expanderHeader { background: rgba(10,25,47,0.6) !important; border: 1px solid rgba(0,212,170,0.12) !important; border-radius: 10px !important; color: var(--text-primary) !important; }
.streamlit-expanderContent { background: rgba(2,12,24,0.6) !important; border: 1px solid rgba(0,212,170,0.08) !important; border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 20px !important; }
[data-testid="stFileUploader"] { background: rgba(10,25,47,0.4) !important; border: 1px dashed rgba(0,212,170,0.3) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NAVAL PHYSICS CONSTANTS & SVGs
# ═══════════════════════════════════════════════════════════════════════════════
VESSEL_LIGHTWEIGHT = 12000.0  # Empty steel weight (MT)
BALLAST_WATER_MT   = 18000.0  # Assumed ballast when Cargo = 0
HOTEL_LOAD_MT      = 2.5      # Baseline auxiliary generator burn per day
GHOST_TOL_MT       = -15.0    # Forgives physical sounding tape slosh in rough seas
CO2_FACTOR         = 3.114   
IQR_MULT           = 1.5     

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified',
    hoverlabel=dict(bgcolor='rgba(10, 25, 47, 0.95)', bordercolor='rgba(0,212,170,0.4)', font=dict(family='Space Mono', size=12, color='#e8f4fd')),
    font=dict(family='DM Sans', color='#5c7c9c', size=11), title_font=dict(family='Syne', color='#ffffff', size=14),
    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='rgba(255,255,255,0.1)', tickfont=dict(family='Space Mono', size=10, color='#5c7c9c'), showspikes=True, spikemode='across', spikethickness=1, spikecolor='rgba(255,255,255,0.2)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, tickfont=dict(family='Space Mono', size=10, color='#5c7c9c')),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)', font=dict(family='Space Mono', size=10, color='#8aa8c4')),
    margin=dict(l=10, r=10, t=50, b=10),
)

TEAL, GOLD, RED, BLUE = '#00d4aa', '#f5c842', '#ff4560', '#008ffb'

def get_uri(svg_str): return f"data:image/svg+xml;base64,{base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')}"

SVG_ICONS = {
    'VERIFIED': get_uri('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="6" fill="#00e396"><animate attributeName="opacity" values="1;0.3;1" dur="2.5s" repeatCount="indefinite"/></circle></svg>'),
    'GHOST BUNKER': get_uri('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="#ff4560"><animate attributeName="opacity" values="1;0;1" dur="0.7s" repeatCount="indefinite"/></circle><path d="M12 7v6m0 4v.01" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></svg>'),
    'STAT OUTLIER': get_uri('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4" fill="#ff4560"><animate attributeName="r" values="4;11;4" dur="1.5s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0;1" dur="1.5s" repeatCount="indefinite"/></circle></svg>'),
    'LEDGER VARIANCE': get_uri('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><polygon points="12,3 3,20 21,20" fill="#f5a623"><animate attributeName="opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite"/></polygon></svg>')
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLOSED-LOOP THERMODYNAMIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def parse_datetime(d_val, t_val) -> pd.Timestamp:
    if isinstance(d_val, pd.Timestamp): d_str = d_val.strftime('%Y-%m-%d')
    else:
        d_str = str(d_val).strip()
        d_str = re.sub(r'20224', '2024', d_str)
        d_str = re.sub(r'20023', '2023', d_str)
        d_str = re.sub(r'(\d+)\s+([A-Za-z]+)\.?\s+(\d{4})', lambda m: f"{m.group(3)}-{m.group(2)[:3].upper()}-{m.group(1).zfill(2)}", d_str)
        try: d_str = pd.to_datetime(d_str, errors='coerce').strftime('%Y-%m-%d')
        except: pass
    if isinstance(t_val, pd.Timestamp): t_str = t_val.strftime('%H:%M')
    else:
        t_raw = str(t_val).upper().strip().replace('H', '').replace('LT', '').replace('UTC', '').replace(' ', '')
        m = re.match(r'^(\d{1,2}):(\d{2})', t_raw)
        if m: t_str = f"{m.group(1).zfill(2)}:{m.group(2)}"
        elif re.match(r'^\d{4}$', t_raw): t_str = f"{t_raw[:2]}:{t_raw[2:]}"
        elif re.match(r'^\d{3}$', t_raw): t_str = f"0{t_raw[0]}:{t_raw[1:]}"
        elif re.match(r'^\d{1,2}$', t_raw): t_str = f"{t_raw.zfill(2)}:00"
        else: t_str = "00:00"
    try: return pd.to_datetime(f"{d_str} {t_str}", errors='coerce')
    except: return pd.NaT

def process_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    v_name = re.sub(r'\.[^.]+$', '', uploaded_file.name).upper()
    if uploaded_file.name.lower().endswith('.xlsx'): df_raw = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
    else: df_raw  = pd.read_csv(io.StringIO(uploaded_file.read().decode('latin-1', errors='replace')), sep=None, engine='python', header=None, on_bad_lines='skip')

    # Dynamically locate header
    keywords = ['PORT', 'VOY', 'DATE', 'A/D', 'FO', 'ROB', 'DIST']
    best_row, max_score = 0, 0
    for i in range(min(60, len(df_raw))):
        score = sum(1 for k in keywords if k in ' '.join(df_raw.iloc[i].fillna('').astype(str).str.upper()))
        if score > max_score: max_score, best_row = score, i

    h0 = df_raw.iloc[best_row].fillna('').astype(str).str.strip().str.upper()
    h1 = df_raw.iloc[best_row+1].fillna('').astype(str).str.strip().str.upper() if best_row+1 < len(df_raw) else pd.Series([''] * len(h0))
    df = df_raw.iloc[best_row+2:].copy().reset_index(drop=True)
    df.columns = (h0.replace('', np.nan).ffill().fillna('') + '_' + h1).str.strip('_')[:len(df.columns)]

    cmap = {}
    for c in df.columns:
        u = str(c).upper().replace(' ', '')
        if 'DATE' in u and 'Date' not in cmap.values(): cmap[c] = 'Date'
        elif 'TIME' in u and 'TOTAL' not in u and 'Time' not in cmap.values(): cmap[c] = 'Time'
        elif 'PORT' in u or 'LOCATION' in u and 'Port' not in cmap.values(): cmap[c] = 'Port'
        elif 'VOY' in u and 'Voy' not in cmap.values(): cmap[c] = 'Voy'
        elif 'A/D' in u or 'ARR' in u or 'DEP' in u and 'AD' not in cmap.values(): cmap[c] = 'AD'
        
        # Bridge Kinematics
        elif 'TOTALDIST' in u and 'TotalDist' not in cmap.values(): cmap[c] = 'TotalDist'
        elif 'TOTALTIME' in u and 'TotalTime' not in cmap.values(): cmap[c] = 'TotalTime'
        elif 'SPEED' in u and 'Speed' not in cmap.values(): cmap[c] = 'Speed'
        
        # Bunkers & ROB
        elif ('BUNKER' in u or 'RECEIV' in u) and ('FO' in u or 'VLSFO' in u) and 'Bunk_FO' not in cmap.values(): cmap[c] = 'Bunk_FO'
        elif ('BUNKER' in u or 'RECEIV' in u) and 'MGO' in u and 'Bunk_MGO' not in cmap.values(): cmap[c] = 'Bunk_MGO'
        elif ('ROB' in u and 'FOA' in u) or u.endswith('FOA') and 'FO_A' not in cmap.values(): cmap[c] = 'FO_A'
        elif ('ROB' in u and 'FOL' in u) or u.endswith('FOL') and 'FO_L' not in cmap.values(): cmap[c] = 'FO_L'
        elif ('ROB' in u and 'MGOA' in u) or u.endswith('MGOA') and 'MGO_A' not in cmap.values(): cmap[c] = 'MGO_A'
        
        # Cargo weight
        elif 'QUANTITY' in u and 'Qty' not in cmap.values(): cmap[c] = 'Qty'

    df = df.rename(columns=cmap)
    
    # Ensure critical columns exist
    for col in ['Date','Time','Port','Voy','AD','TotalDist','TotalTime','Speed','Bunk_FO','Bunk_MGO','FO_A','FO_L','MGO_A','Qty']:
        if col not in df.columns: df[col] = 0.0

    df['Datetime'] = df.apply(lambda r: parse_datetime(r['Date'], r['Time']), axis=1)
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    
    for col in ['TotalDist','TotalTime','Speed','Bunk_FO','Bunk_MGO','FO_A','FO_L','MGO_A','Qty']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.\-]','',regex=True), errors='coerce').fillna(0.0)

    df['AD'] = df['AD'].astype(str).str.upper().str.strip()
    df.loc[df['AD'].str.contains('A', na=False), 'AD'] = 'A'
    df.loc[df['AD'].str.contains('D', na=False), 'AD'] = 'D'

    trips = []
    for i in range(len(df) - 1):
        r1, r2 = df.iloc[i], df.iloc[i+1]
        
        if r1['AD'] == 'D' and r2['AD'] == 'A': leg = 'SEA PASSAGE'
        elif r1['AD'] == 'A' and r2['AD'] == 'D': leg = 'PORT STAY'
        else: continue # Skip invalid pairs

        # TIME ANCHORING (Trust ECDIS Time to fix Timezone bugs)
        hours = r2['TotalTime']
        if hours <= 0: hours = (r2['Datetime'] - r1['Datetime']).total_seconds() / 3600
        if hours <= 0: continue
        days = hours / 24.0

        # CLOSED-LOOP FUEL EXTRACTION (Total Fuel Equivalent)
        hfo_consumed = (r1["FO_A"] - r2["FO_A"]) + r2["Bunk_FO"]
        mgo_consumed = max(0.0, (r1["MGO_A"] - r2["MGO_A"]) + r2["Bunk_MGO"])
        total_fuel_mt = hfo_consumed + mgo_consumed
        
        daily_burn = total_fuel_mt / days if days > 0 else 0.0
        
        # PROPULSION ISOLATION & CARGO NORMALIZATION
        propulsion_burn_day = max(0.1, daily_burn - HOTEL_LOAD_MT) if leg == 'SEA PASSAGE' else 0.0
        
        cargo_qty = r2['Qty'] if r2['Qty'] > 0 else r1['Qty']
        displacement = cargo_qty + VESSEL_LIGHTWEIGHT + (BALLAST_WATER_MT if cargo_qty == 0 else 0.0)

        # ADMIRALTY MATHEMATICS
        leg_nm = r2['TotalDist'] if r2['TotalDist'] > 0 else 0.0
        speed = r2['Speed'] if r2['Speed'] > 0 else (leg_nm / hours if hours > 0 else 0.0)
        
        if propulsion_burn_day > 0 and speed > 0 and leg == 'SEA PASSAGE':
            admiralty_coeff = ((displacement ** (2/3)) * (speed ** 3)) / propulsion_burn_day
        else:
            admiralty_coeff = 0.0

        # INTEGRITY FLAGS
        gap_fo = abs(r2['FO_A'] - r2['FO_L'])
        dynamic_ledger_tol = max(30.0, 0.03 * r2['FO_A']) # 3% or 30 MT tolerance
        
        status = 'VERIFIED'
        if total_fuel_mt < GHOST_TOL_MT: 
            status = 'GHOST BUNKER'
            total_fuel_mt, daily_burn, admiralty_coeff = 0.0, 0.0, 0.0
        elif gap_fo > dynamic_ledger_tol: 
            status = 'LEDGER VARIANCE'

        trips.append({
            'Timeline': f"{r1['Datetime'].strftime('%b %d')} → {r2['Datetime'].strftime('%b %d')}",
            'Route': f"{str(r1['Port']).strip()[:15]} → {str(r2['Port']).strip()[:15]}",
            'Leg': leg, 'Voy': str(r1['Voy']).strip(), 'Days': round(days, 2),
            'Speed': round(speed, 2), 'Total_Fuel_MT': round(total_fuel_mt, 2), 'Daily_Burn': round(daily_burn, 2),
            'CO2_MT': round(total_fuel_mt * CO2_FACTOR, 2), 'Admiralty': round(admiralty_coeff, 0),
            'Status': status, 'Indicator': SVG_ICONS.get(status, SVG_ICONS['VERIFIED']),
            '_raw_adm': admiralty_coeff # Hidden strictly for stats
        })

    trip_df = pd.DataFrame(trips)
    if trip_df.empty: return trip_df, v_name

    # ADVANCED IQR ON ADMIRALTY COEFFICIENT
    sea_verified = (trip_df['Leg'] == 'SEA PASSAGE') & (trip_df['Status'] == 'VERIFIED') & (trip_df['_raw_adm'] > 0)
    if sea_verified.sum() > 4:
        q1, q3 = trip_df.loc[sea_verified, '_raw_adm'].quantile(0.25), trip_df.loc[sea_verified, '_raw_adm'].quantile(0.75)
        iqr = q3 - q1
        outlier_mask = sea_verified & ((trip_df['_raw_adm'] < q1 - IQR_MULT * iqr) | (trip_df['_raw_adm'] > q3 + IQR_MULT * iqr))
        
        trip_df.loc[outlier_mask, 'Status'] = 'STAT OUTLIER'
        trip_df.loc[outlier_mask, 'Indicator'] = SVG_ICONS['STAT OUTLIER']

    return trip_df.drop(columns=['_raw_adm']), v_name

def compute_vessel_stats(df: pd.DataFrame) -> dict:
    sea = df[(df['Leg'] == 'SEA PASSAGE') & (df['Status'] == 'VERIFIED')]
    n_sea = len(sea)
    return {
        'total_fuel': round(df['Total_Fuel_MT'].sum(), 1), 'total_co2': round(df['CO2_MT'].sum(), 1),
        'anomalies': len(df[df['Status'] != 'VERIFIED']),
        'integrity': round(len(df[df['Status'] == 'VERIFIED']) / len(df) * 100, 1) if len(df) > 0 else 0,
        'admiralty_avg': round(sea['Admiralty'].mean(), 0) if n_sea > 0 else 0
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
def chart_fuel_timeline(df: pd.DataFrame) -> go.Figure:
    sea, port, anom = df[df['Leg'] == 'SEA PASSAGE'], df[df['Leg'] == 'PORT STAY'], df[df['Status'] != 'VERIFIED']
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    
    fig.add_trace(go.Bar(x=sea['Timeline'], y=sea['Total_Fuel_MT'], name='Sea Burn', marker_color=TEAL, marker_line_width=0, opacity=0.85, hovertemplate='Fuel: %{y:.1f} MT<br>Speed: %{customdata[0]:.1f} kn<extra></extra>', customdata=sea[['Speed']].values), row=1, col=1)
    fig.add_trace(go.Bar(x=port['Timeline'], y=port['Total_Fuel_MT'], name='Port Burn', marker_color=BLUE, marker_line_width=0, opacity=0.6), row=1, col=1)
    if not anom.empty: fig.add_trace(go.Scatter(x=anom['Timeline'], y=anom['Total_Fuel_MT'], mode='markers', name='Anomaly', marker=dict(symbol='x', size=10, color=RED), hovertemplate='⚠ %{customdata}<extra></extra>', customdata=anom['Status'].values), row=1, col=1)
    fig.add_trace(go.Scatter(x=sea['Timeline'], y=sea['Speed'], mode='lines+markers', name='Speed', line=dict(color=GOLD, width=2.5, shape='spline'), marker=dict(size=4)), row=2, col=1)

    layout = dict(**PLOTLY_LAYOUT)
    xa2, ya2 = PLOTLY_LAYOUT['xaxis'].copy(), PLOTLY_LAYOUT['yaxis'].copy()
    xa2.update({'tickangle': -35}); ya2.update({'title': 'kn'})
    layout.update({'height': 400, 'barmode': 'overlay', 'xaxis': dict(showticklabels=False), 'xaxis2': xa2, 'yaxis2': ya2})
    fig.update_layout(**layout)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN UI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div style="position:relative; z-index:2;">
    <p class="kpi-label" style="color: #00d4aa; letter-spacing: 0.2em;">POSEIDON INTELLIGENCE ENGINE  ·  v5.0</p>
    <h1 class="hero-title">Fleet Command Center</h1>
    <p style="color:#6b8cae; font-size:0.9rem; margin:0; max-width:700px;">
      <b>Closed-Loop Auditing:</b> ECDIS Time Anchoring · Total Fuel Equivalent (HFO+MGO) · 
      Propulsion Isolation · Dynamic Ballast Normalization.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Fleet Excel/CSV Logs", accept_multiple_files=True, type=['csv', 'xlsx'], label_visibility="collapsed")

if not uploaded_files: st.stop()

vessel_data = {}
for f in uploaded_files:
    try:
        df, v_name = process_file(f)
        if not df.empty: vessel_data[v_name] = (df, compute_vessel_stats(df))
    except Exception as e:
        st.error(f"Could not parse {f.name}: {str(e)}")

if not vessel_data: st.error("No valid telemetry extracted."); st.stop()

st.markdown("<div class='section-header'><div class='section-line'></div><div class='section-title'>TACTICAL DOSSIERS</div><div class='section-line' style='background:linear-gradient(270deg,rgba(0,212,170,0.3),transparent)'></div></div>", unsafe_allow_html=True)

for v_name, (df, st_s) in vessel_data.items():
    with st.expander(f"⚓  {v_name}  —  {st_s['integrity']}% INTEGRITY", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        
        int_color = '#00e396' if st_s['integrity'] >= 80 else ('#f5c842' if st_s['integrity'] >= 50 else '#ff4560')
        c1.markdown(f"<div class='kpi-card' style='--accent-color:{int_color};'><div class='kpi-label'>DATA INTEGRITY</div><div class='kpi-value'>{st_s['integrity']}%</div><div class='kpi-delta'>Confidence Score</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card' style='--accent-color:#f5c842;'><div class='kpi-label'>ADMIRALTY AVG</div><div class='kpi-value'>{st_s['admiralty_avg']:.0f}</div><div class='kpi-delta'>Cargo Normalized</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card' style='--accent-color:#008ffb;'><div class='kpi-label'>TOTAL FUEL</div><div class='kpi-value'>{st_s['total_fuel']:,.0f}</div><div class='kpi-delta'>MT Consumed (HFO+MGO)</div></div>", unsafe_allow_html=True)
        color = '#ff4560' if st_s['anomalies'] > 0 else '#00e396'
        c4.markdown(f"<div class='kpi-card' style='--accent-color:{color};'><div class='kpi-label'>QUARANTINED</div><div class='kpi-value'>{st_s['anomalies']}</div><div class='kpi-delta'>Anomalies Blocked</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2 = st.tabs(['📋 CLOSED-LOOP MATRIX', '📈 THERMODYNAMIC KINEMATICS'])

        with t1:
            safe_max_burn = max(float(df['Daily_Burn'].max() if not df.empty else 0) * 1.05, 1.0)
            
            def highlight(row):
                return ['background-color: rgba(255, 69, 96, 0.12); color: #ff4560;' if row['Status'] != 'VERIFIED' else ''] * len(row)

            display_df = df.copy()
            styled_df = display_df.style.apply(highlight, axis=1).format({'Days': '{:.2f}', 'Speed': '{:.1f}', 'Total_Fuel_MT': '{:.1f}', 'Daily_Burn': '{:.1f}', 'Admiralty': '{:.0f}'})

            st.dataframe(
                styled_df,
                column_config={
                    'Timeline': st.column_config.TextColumn('TIMELINE', width='medium'),
                    'Route': st.column_config.TextColumn('ROUTE', width='large'),
                    'Leg': st.column_config.TextColumn('PHASE'),
                    'Speed': st.column_config.NumberColumn('SPEED (kn)'),
                    'Total_Fuel_MT': st.column_config.NumberColumn('TOTAL FUEL', help="HFO + MGO"),
                    'Daily_Burn': st.column_config.ProgressColumn('BURN (MT/d)', format='%.1f', min_value=0, max_value=safe_max_burn),
                    'Admiralty': st.column_config.NumberColumn('ADMIRALTY', help='Propulsion isolated, ballast normalized'),
                    'Indicator': st.column_config.ImageColumn('⚡', width='small'),
                    'Status': st.column_config.TextColumn('STATUS', width='medium'),
                    'Voy': None, 'Days': None, 'CO2_MT': None
                },
                hide_index=True, use_container_width=True, height=450
            )
            
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w: display_df.drop(columns=['Indicator'], errors='ignore').to_excel(w, index=False)
            buf.seek(0)
            st.download_button("📥 EXPORT SECURE LEDGER", data=buf, file_name=f"{v_name}_LEDGER.xlsx", key=f"dl_{v_name}")

        with t2: st.plotly_chart(chart_fuel_timeline(df), use_container_width=True, config={'displayModeBar': False})
