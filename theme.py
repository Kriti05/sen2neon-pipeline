"""
theme.py — SEN2NEON "Aurora Nebula" design system v3
Bulletproof against Streamlit 1.30+ DOM changes.
"""
from __future__ import annotations

import streamlit as st

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#F7F8FC"
PANEL       = "#FFFFFF"
INK         = "#0F172A"
MUTED       = "#64748B"
BORDER      = "#E7EBF3"
OBSIDIAN    = "#0A0E1A"
OBSIDIAN_2  = "#141B2E"
VIOLET      = "#7C3AED"
VIOLET_DARK = "#6D28D9"
CYAN        = "#06B6D4"
AMBER       = "#F59E0B"
DANGER      = "#EF4444"

SIDEBAR_MUTED = "#8B98B8"

# Chart colors
LINE_A, LINE_B = VIOLET, CYAN
BAR_BASE       = "#C4B5FD"
BAR_BEST       = VIOLET
BAR_WORST      = DANGER


def style_fig(fig, height: int = 400):
    """Apply Aurora Nebula styling to a Plotly figure."""
    fig.update_layout(
        autosize=True,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=INK,
            family="Inter, Plus Jakarta Sans, system-ui, sans-serif",
            size=12,
        ),
        margin=dict(t=28, b=44, l=48, r=20),
        legend=dict(
            orientation="h",
            y=1.14,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11.5),
        ),
        transition={"duration": 320, "easing": "cubic-in-out"},
    )
    fig.update_xaxes(
        gridcolor="rgba(15, 23, 42, 0.06)",
        automargin=True,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(15, 23, 42, 0.06)",
        automargin=True,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
    )
    return fig


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════
   TOKENS
   ═══════════════════════════════════════════════════════════════════════════ */
:root {
  --bg: #F7F8FC;
  --panel: #FFFFFF;
  --ink: #0F172A;
  --ink-soft: #1E293B;
  --muted: #64748B;
  --border: #E7EBF3;
  --border-strong: #CBD5E1;

  --obsidian: #0A0E1A;
  --obsidian-2: #141B2E;

  --violet: #7C3AED;
  --violet-dark: #6D28D9;
  --violet-light: #A78BFA;
  --cyan: #06B6D4;
  --amber: #F59E0B;
  --danger: #EF4444;
  --success: #10B981;

  --r-sm: 6px;
  --r: 8px;
  --r-lg: 12px;
  --r-xl: 16px;
  --gap: 12px;
  --gap-sm: 8px;
  --pad: clamp(0.9rem, 2.4vw, 2rem);

  --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --dur: 0.22s;

  --sh-xs: 0 1px 2px rgba(15,23,42,.04);
  --sh-sm: 0 2px 8px rgba(15,23,42,.05);
  --sh-md: 0 8px 24px rgba(15,23,42,.08);
  --sh-lg: 0 20px 48px rgba(15,23,42,.12);
  --sh-violet: 0 8px 24px rgba(124,58,237,.28);
}

/* ═══════════════════════════════════════════════════════════════════════════
   BASE — force light main, dark sidebar
   ═══════════════════════════════════════════════════════════════════════════ */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
  font-family: "Inter", "Plus Jakarta Sans", system-ui, -apple-system, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main {
  background: var(--bg) !important;
  color: var(--ink) !important;
}

header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.block-container {
  max-width: 1400px;
  padding: var(--pad) var(--pad) 3rem !important;
}

h1, h2, h3, h4 {
  font-family: "Plus Jakarta Sans", "Inter", sans-serif !important;
  letter-spacing: -0.025em;
  color: var(--ink) !important;
  font-weight: 800;
}

a { color: var(--violet); text-decoration: none; transition: color var(--dur) var(--ease); }
a:hover { color: var(--violet-dark); }

/* Force dark ink on ALL main text */
[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label,
[data-testid="stMain"] div,
[data-testid="stMain"] li {
  color: var(--ink);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SPACING
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stVerticalBlock"] { gap: var(--gap) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { gap: var(--gap) !important; }
div[data-testid="stHorizontalBlock"] {
  gap: var(--gap) !important;
  flex-wrap: wrap;
}
div[data-testid="stMarkdownContainer"] p { margin: 0 0 var(--gap-sm); }

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR — obsidian, 100% width nav
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--obsidian) 0%, var(--obsidian-2) 100%) !important;
  border-right: 1px solid rgba(148, 163, 184, 0.08) !important;
}
section[data-testid="stSidebar"] * { color: #E2E8F5 !important; }
section[data-testid="stSidebar"] hr {
  border-color: rgba(148, 163, 184, 0.14);
  margin: 18px 0;
}
section[data-testid="stSidebar"] div[data-testid="stAlert"] * { color: var(--ink) !important; }

/* NAV RADIO — force flex column, full width, no circle */
section[data-testid="stSidebar"] div[role="radiogroup"] {
  display: flex !important;
  flex-direction: column !important;
  gap: 6px !important;
  width: 100% !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  display: flex !important;
  align-items: center !important;
  width: 100% !important;
  min-height: 46px !important;
  padding: 12px 14px !important;
  margin: 0 !important;
  border-radius: var(--r) !important;
  border: 1px solid transparent !important;
  border-left: 3px solid transparent !important;
  background: rgba(255, 255, 255, 0.03) !important;
  cursor: pointer !important;
  box-sizing: border-box !important;
  transition:
    background var(--dur) var(--ease),
    border-color var(--dur) var(--ease),
    transform var(--dur) var(--ease) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  background: rgba(255, 255, 255, 0.07) !important;
  border-color: rgba(124, 58, 237, 0.3) !important;
  transform: translateX(3px) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(90deg, rgba(124, 58, 237, 0.32), rgba(6, 182, 212, 0.14)) !important;
  border-color: rgba(124, 58, 237, 0.5) !important;
  border-left-color: var(--amber) !important;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.22) !important;
}

/* Hide radio circle — every possible selector */
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child,
section[data-testid="stSidebar"] div[role="radiogroup"] > label > span:first-child,
section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child {
  width: 100% !important;
  flex: 1 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
  margin: 0 !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  letter-spacing: -0.01em !important;
  width: 100% !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN RADIO (segmented control)
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMain"] div[role="radiogroup"] {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}

[data-testid="stMain"] div[role="radiogroup"] > label {
  background: var(--panel) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 9px 18px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  cursor: pointer !important;
  transition: all var(--dur) var(--ease) !important;
}

[data-testid="stMain"] div[role="radiogroup"] > label:hover {
  border-color: var(--violet-light) !important;
  transform: translateY(-1px);
  box-shadow: var(--sh-xs);
}

[data-testid="stMain"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(135deg, var(--obsidian), var(--obsidian-2)) !important;
  border-color: var(--obsidian) !important;
  box-shadow: 0 4px 14px rgba(10, 14, 26, 0.22) !important;
}

[data-testid="stMain"] div[role="radiogroup"] > label:has(input:checked) * {
  color: #FFFFFF !important;
}

[data-testid="stMain"] div[role="radiogroup"] > label > div:first-child {
  display: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════════════════ */
.stButton > button,
.stDownloadButton > button {
  min-height: 48px;
  border-radius: var(--r) !important;
  font-weight: 700;
  width: 100%;
  font-size: 0.94rem;
  transition: all var(--dur) var(--ease) !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--violet) 0%, var(--violet-dark) 100%) !important;
  color: #fff !important;
  border: 0 !important;
  box-shadow: var(--sh-violet);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 32px rgba(124, 58, 237, 0.38) !important;
}
.stButton > button:active { transform: translateY(0); }
.stButton > button p { color: #fff !important; }

.stDownloadButton > button {
  background: var(--panel) !important;
  color: var(--violet) !important;
  border: 1.5px solid var(--border) !important;
}
.stDownloadButton > button:hover {
  border-color: var(--violet) !important;
  background: #F5F3FF !important;
  transform: translateY(-2px);
}
.stDownloadButton > button p { color: var(--violet) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   METRICS
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: clamp(14px, 1.6vw, 20px) !important;
  height: 100%;
  box-shadow: var(--sh-sm);
  position: relative;
  overflow: hidden;
  transition: all var(--dur) var(--ease);
}
div[data-testid="stMetric"]::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--violet), var(--cyan));
}
div[data-testid="stMetric"]:hover {
  border-color: rgba(124, 58, 237, 0.3) !important;
  box-shadow: var(--sh-md);
  transform: translateY(-3px);
}
div[data-testid="stMetricLabel"] p {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] > div {
  color: var(--ink) !important;
  font-family: "Plus Jakarta Sans", sans-serif !important;
  font-weight: 800;
  font-size: clamp(1.2rem, 2.3vw, 1.8rem) !important;
  line-height: 1.15;
}

/* ═══════════════════════════════════════════════════════════════════════════
   FILE UPLOADER
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] { width: 100% !important; }
[data-testid="stFileUploader"] > label {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--panel) !important;
  border: 2px dashed var(--border-strong) !important;
  border-radius: var(--r-lg) !important;
  padding: 20px !important;
  width: max-content;
  transition: all var(--dur) var(--ease) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--violet) !important;
  background: #F5F3FF !important;
  box-shadow: 0 0 0 5px rgba(124, 58, 237, 0.10) !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background: linear-gradient(135deg, var(--violet), var(--violet-dark)) !important;
  color: #fff !important;
  border-radius: var(--r) !important;
  padding: 8px 18px !important;
  font-weight: 600 !important;
  border: 0 !important;
  min-height: auto !important;
  width: auto !important;
}
[data-testid="stFileUploaderDropzone"] button p { color: #fff !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   INPUTS / SELECT / TEXTAREA
   ═══════════════════════════════════════════════════════════════════════════ */
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="base-input"],
textarea {
  border-radius: var(--r) !important;
  border-color: var(--border) !important;
  background: var(--panel) !important;
  color: var(--ink) !important;
}
[data-baseweb="select"] > div:hover,
[data-baseweb="input"]:hover {
  border-color: var(--violet-light) !important;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"]:focus-within {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.14) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ALERTS / EXPANDERS
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stAlert"] {
  border-radius: var(--r) !important;
  border: 1px solid var(--border) !important;
}
details {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  overflow: hidden;
  transition: all var(--dur) var(--ease);
}
details:hover {
  border-color: rgba(124, 58, 237, 0.3) !important;
  box-shadow: var(--sh-sm);
}
details summary { padding: 12px 16px !important; font-weight: 600; }

/* ═══════════════════════════════════════════════════════════════════════════
   IMAGES / TABLES / CHARTS
   ═══════════════════════════════════════════════════════════════════════════ */
.stImage img {
  border-radius: var(--r-lg);
  border: 1px solid var(--border);
  max-width: 100%;
  height: auto;
  box-shadow: var(--sh-sm);
  transition: all var(--dur) var(--ease);
}
.stImage img:hover {
  box-shadow: var(--sh-md);
  transform: translateY(-2px);
}
.stDataFrame,
div[data-testid="stTable"] {
  border-radius: var(--r-lg);
  overflow-x: auto;
  border: 1px solid var(--border);
  box-shadow: var(--sh-xs);
}
.js-plotly-plot, .plotly { max-width: 100% !important; }
div[data-testid="stPlotlyChart"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 12px !important;
  box-shadow: var(--sh-sm);
  transition: box-shadow var(--dur) var(--ease);
}
div[data-testid="stPlotlyChart"]:hover { box-shadow: var(--sh-md); }

/* ═══════════════════════════════════════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════════════════════════════════════ */
.hero {
  position: relative;
  overflow: hidden;
  border-radius: var(--r-xl);
  padding: clamp(24px, 4vw, 52px);
  background:
    radial-gradient(900px 400px at 100% 0%, rgba(124, 58, 237, 0.45), transparent 65%),
    radial-gradient(700px 380px at 0% 100%, rgba(6, 182, 212, 0.28), transparent 60%),
    linear-gradient(135deg, var(--obsidian) 0%, var(--obsidian-2) 100%);
  box-shadow: var(--sh-lg);
  margin-bottom: 8px;
  animation: heroIn 0.5s var(--ease) both;
}
@keyframes heroIn {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
.hero::after {
  content: "";
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--amber), var(--violet), var(--cyan));
}
.hero h1 {
  position: relative;
  color: #fff !important;
  font-size: clamp(1.7rem, 4.4vw, 3rem);
  font-weight: 800;
  line-height: 1.08;
  margin: 0 0 14px;
  max-width: 22ch;
}
.hero p {
  position: relative;
  color: #A8B4D4 !important;
  margin: 0;
  line-height: 1.7;
  max-width: 66ch;
  font-size: clamp(0.92rem, 1.45vw, 1.06rem);
}
.hero .pill {
  position: relative;
  display: inline-block;
  padding: 6px 13px;
  margin: 18px 6px 0 0;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: rgba(124, 58, 237, 0.22);
  color: #DDD6FE !important;
  border: 1px solid rgba(167, 139, 250, 0.4);
  transition: all var(--dur) var(--ease);
}
.hero .pill:hover {
  transform: translateY(-2px);
  border-color: rgba(167, 139, 250, 0.85);
}
.hero .pill.green {
  background: rgba(6, 182, 212, 0.18);
  color: #A5F3FC !important;
  border-color: rgba(6, 182, 212, 0.5);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION HEADER
   ═══════════════════════════════════════════════════════════════════════════ */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 20px 0 6px;
  padding: 0 0 8px;
  border-bottom: 1px solid var(--border);
}
.section-header .bar {
  width: 4px;
  height: 20px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--amber), var(--violet), var(--cyan));
  flex: none;
}
.section-header h3 {
  margin: 0 !important;
  font-size: clamp(1.02rem, 1.9vw, 1.18rem) !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  color: var(--ink) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROGRESS
   ═══════════════════════════════════════════════════════════════════════════ */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
  border-radius: 999px;
}
.stProgress > div > div {
  border-radius: 999px;
  background: var(--border) !important;
  height: 8px !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   FOCUS / SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */
button:focus-visible,
[role="radio"]:focus-visible,
input:focus-visible,
summary:focus-visible {
  outline: 3px solid var(--cyan) !important;
  outline-offset: 2px;
  border-radius: var(--r);
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--violet-light), var(--cyan));
  border-radius: 999px;
  border: 2px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, var(--violet), var(--cyan));
}
::-webkit-scrollbar-track { background: transparent; }

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE ENTER ANIMATION
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMain"] > div {
  animation: fadeUp 0.4s var(--ease) both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  min-width: min(100%, 160px);
}

@media (max-width: 1024px) {
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 calc(33.333% - var(--gap)) !important;
    min-width: 180px;
  }
}

@media (max-width: 768px) {
  :root { --pad: 0.9rem; --gap: 10px; }

  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 calc(50% - var(--gap)) !important;
    min-width: 150px;
  }
  div[data-testid="stHorizontalBlock"]:has(.stImage) > div,
  div[data-testid="stHorizontalBlock"]:has(.stButton) > div,
  div[data-testid="stHorizontalBlock"]:has(.stDownloadButton) > div,
  div[data-testid="stHorizontalBlock"]:has(section[data-testid="stFileUploaderDropzone"]) > div {
    flex: 1 1 100% !important;
  }
  .hero { padding: clamp(20px, 5vw, 32px); border-radius: var(--r-lg); }
  .hero h1 { max-width: 100%; }
  .section-header { margin-top: 14px; }
}

@media (max-width: 420px) {
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 100% !important;
  }
  [data-testid="stMain"] div[role="radiogroup"] > label {
    flex: 1 1 100%;
    text-align: center;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] > label { min-height: 48px; }
  .hero .pill { margin-top: 12px; font-size: 0.65rem; }
  .hero h1 { font-size: 1.5rem; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition: none !important;
    animation: none !important;
  }
}
</style>
"""


def apply_theme() -> None:
    """Inject the Aurora Nebula design system."""
    st.markdown(_CSS, unsafe_allow_html=True)