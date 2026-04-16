# config.py
import base64

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS & BUSINESS RULES
# ═══════════════════════════════════════════════════════════════════════════════
MIN_SEA_SPEED_KN = 2.0
MIN_SEA_DAYS = 0.05
GHOST_BUNKER_TOLERANCE_MT = -2.0
PORT_GHOST_BUNKER_TOLERANCE_MT = -5.0

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTRACTS (SCHEMA ENFORCEMENT)
# ═══════════════════════════════════════════════════════════════════════════════
REQUIRED_RAW_COLS = [
    'FO_A', 'FO_L', 'MGO_A', 'MGO_L', 
    'Bunk_FO', 'Bunk_MGO', 'Bunk_MELO', 'Bunk_HSCYLO', 'Bunk_LSCYLO', 'Bunk_GELO', 'Bunk_CYLO', 
    'MELO_R', 'HSCYLO_R', 'LSCYLO_R', 'GELO_R', 'CYLO_R', 
    'Speed', 'DistLeg', 'TotalDist', 'CargoQty', 'Voy', 'Port', 'AD', 'Date', 'Time'
]

# ═══════════════════════════════════════════════════════════════════════════════
# ASSETS & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
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
.stDataFrame{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var
