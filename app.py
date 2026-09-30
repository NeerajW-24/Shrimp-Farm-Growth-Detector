from __future__ import annotations

import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "shrimp dataset.xlsx"
MODEL_DIR = BASE_DIR / "model"
ENGINEERED_FEATURES = {"Temp_DO", "pH_Ammonia"}
TARGET_NAMES = {"length", "weight"}
SUPPORTED_MODEL_SUFFIXES = {".pkl", ".joblib", ".h5", ".keras"}

st.set_page_config(
    page_title="Shrimp Farm Helper",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --ocean-900: #06283d;
    --ocean-700: #0b4f6c;
    --ocean-500: #0fa3b1;
    --ocean-300: #7fd8e0;
    --ocean-100: #e3f6f8;
    --coral: #ff6b57;
    --coral-soft: #ffe3de;
    --sun: #ffb703;
    --mint: #2ec4a6;
    --ink: #0b2a3c;
    --muted: #5b7486;
    --card: rgba(255, 255, 255, 0.86);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4, .hero-card h1, .section-head h2 {
    font-family: 'Poppins', 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(15, 163, 177, 0.18), transparent 60%),
        radial-gradient(700px 480px at 100% 10%, rgba(255, 107, 87, 0.14), transparent 60%),
        radial-gradient(800px 500px at 50% 110%, rgba(127, 216, 224, 0.30), transparent 60%),
        #f3fafc;
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06283d 0%, #0b4f6c 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

/* ---------- Hero ---------- */
.hero-card {
    position: relative;
    overflow: hidden;
    padding: 2.2rem 2.2rem 4.2rem 2.2rem;
    border-radius: 1.75rem;
    background: linear-gradient(135deg, #06283d 0%, #0b4f6c 50%, #0fa3b1 100%);
    box-shadow: 0 24px 50px rgba(6, 40, 61, 0.28);
    color: #ffffff;
}

.hero-card::before {
    content: "";
    position: absolute;
    width: 340px; height: 340px;
    right: -80px; top: -120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 183, 3, 0.45), transparent 65%);
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 260px; height: 260px;
    left: -90px; bottom: -130px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 107, 87, 0.40), transparent 65%);
}

.hero-inner { position: relative; z-index: 2; display: flex; align-items: center; gap: 1.5rem; justify-content: space-between; flex-wrap: wrap; }
.hero-text { flex: 1 1 380px; }

.hero-kicker {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #06283d;
    background: linear-gradient(90deg, #ffb703, #ffd166);
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}

.hero-card h1 {
    font-size: clamp(2rem, 4.4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.08;
    margin: 0 0 0.7rem 0;
    color: #ffffff;
}

.hero-card h1 span {
    background: linear-gradient(90deg, #ffd166, #ff8a7a);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-card p {
    margin: 0;
    font-size: 1.04rem;
    line-height: 1.55;
    color: #d8f1f5;
    max-width: 560px;
}

.hero-art {
    flex: 0 0 auto;
    width: 190px; height: 190px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.35), rgba(255,255,255,0.06));
    border: 1px solid rgba(255, 255, 255, 0.28);
    backdrop-filter: blur(6px);
    font-size: 6.2rem;
    animation: floaty 4.5s ease-in-out infinite;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
}

@keyframes floaty {
    0%, 100% { transform: translateY(0) rotate(-4deg); }
    50% { transform: translateY(-12px) rotate(4deg); }
}

.hero-waves {
    position: absolute; left: 0; right: 0; bottom: -1px; height: 70px; z-index: 1;
}

.step-strip {
    margin-top: 1.3rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}

.step-pill {
    padding: 0.5rem 0.95rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.30);
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: 600;
    backdrop-filter: blur(4px);
    transition: transform .2s ease, background .2s ease;
}
.step-pill:hover { transform: translateY(-2px); background: rgba(255, 255, 255, 0.28); }

/* ---------- Quick stats ---------- */
.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.9rem; margin: 1.1rem 0 0.4rem; }
.stat-tile {
    background: var(--card);
    border: 1px solid rgba(15, 163, 177, 0.18);
    border-radius: 1.1rem;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 10px 24px rgba(11, 79, 108, 0.08);
    display: flex; align-items: center; gap: 0.8rem;
}
.stat-tile .ico { font-size: 1.7rem; }
.stat-tile .num { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.25rem; color: var(--ocean-700); line-height: 1.1; }
.stat-tile .lbl { font-size: 0.8rem; color: var(--muted); }

/* ---------- Section headers ---------- */
.section-head {
    display: flex; align-items: center; gap: 0.9rem;
    margin: 2.1rem 0 0.4rem 0;
}
.section-head .num {
    flex: 0 0 auto;
    width: 2.5rem; height: 2.5rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 0.85rem;
    font-family: 'Poppins', sans-serif; font-weight: 700; color: #fff;
    background: linear-gradient(135deg, var(--coral), #ff9a5a);
    box-shadow: 0 8px 18px rgba(255, 107, 87, 0.35);
}
.section-head h2 { margin: 0; padding: 0; font-size: 1.45rem; font-weight: 700; color: var(--ocean-900); }
.section-sub { color: var(--muted); margin: 0 0 0.9rem 3.4rem; font-size: 0.95rem; }

/* ---------- Cards & metrics ---------- */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid rgba(15, 163, 177, 0.18);
    border-top: 5px solid var(--ocean-500);
    border-radius: 1.1rem;
    padding: 1rem 1.2rem;
    box-shadow: 0 12px 28px rgba(11, 79, 108, 0.10);
    transition: transform .2s ease, box-shadow .2s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow: 0 18px 34px rgba(11, 79, 108, 0.16); }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
div[data-testid="stMetricValue"] { font-family: 'Poppins', sans-serif; font-weight: 700; color: var(--ocean-900); }

div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { border-top-color: var(--sun); }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { border-top-color: var(--coral); }

.small-note { color: var(--muted); font-size: 0.92rem; }

.value-badge {
    margin: 0.2rem 0 0.55rem 0;
    display: inline-block;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: var(--ocean-100);
    color: var(--ocean-700);
    border: 1px solid rgba(15, 163, 177, 0.35);
    font-size: 0.82rem;
    font-weight: 700;
}

.section-card {
    padding: 1rem 1.1rem;
    border-radius: 1.2rem;
    background: var(--card);
    border: 1px solid rgba(15, 163, 177, 0.18);
    box-shadow: 0 12px 28px rgba(11, 79, 108, 0.10);
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-radius: 0.8rem !important;
    border-color: rgba(15, 163, 177, 0.4) !important;
}
div[data-baseweb="input"] input { color: var(--ink) !important; }
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"]:focus-within {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px rgba(255, 107, 87, 0.18) !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background: var(--coral) !important;
    border: 3px solid #fff !important;
    box-shadow: 0 4px 12px rgba(255, 107, 87, 0.5) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child { background: rgba(15, 163, 177, 0.18) !important; }

div[data-testid="stCheckbox"] label span { color: var(--ink) !important; font-weight: 500; }

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] { border-radius: 1rem; border: none; box-shadow: 0 8px 20px rgba(11, 79, 108, 0.08); }

/* ---------- Tables, expanders ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 1rem;
    overflow: hidden;
    border: 1px solid rgba(15, 163, 177, 0.2);
    box-shadow: 0 10px 24px rgba(11, 79, 108, 0.08);
}
div[data-testid="stExpander"] {
    border-radius: 1rem;
    border: 1px solid rgba(15, 163, 177, 0.2);
    background: var(--card);
    box-shadow: 0 8px 20px rgba(11, 79, 108, 0.06);
}
div[data-testid="stExpander"] summary { font-weight: 600; color: var(--ocean-700); }

/* ---------- Footer ---------- */
.app-footer {
    margin-top: 3rem; padding: 1.2rem; text-align: center;
    border-radius: 1.2rem;
    background: linear-gradient(90deg, rgba(15,163,177,0.12), rgba(255,107,87,0.12));
    color: var(--muted); font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def section_header(number: int | str, title: str, subtitle: str = "") -> None:
    sub = f"<p class='section-sub'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='section-head'><div class='num'>{number}</div><h2>{title}</h2></div>{sub}",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
/* ===== Extra motion & polish ===== */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: linear-gradient(120deg, rgba(15,163,177,0.10), rgba(255,183,3,0.08), rgba(255,107,87,0.10), rgba(46,196,166,0.10));
    background-size: 400% 400%;
    animation: auroraShift 18s ease-in-out infinite;
}
@keyframes auroraShift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }

/* rising bubbles */
.bubbles { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.bubbles span {
    position: absolute; bottom: -60px; display: block;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.9), rgba(127,216,224,0.35));
    border: 1px solid rgba(15,163,177,0.25);
    animation: rise linear infinite;
}
.bubbles span:nth-child(1) { left: 6%;  width: 18px; height: 18px; animation-duration: 14s; animation-delay: 0s; }
.bubbles span:nth-child(2) { left: 16%; width: 30px; height: 30px; animation-duration: 19s; animation-delay: 3s; }
.bubbles span:nth-child(3) { left: 28%; width: 12px; height: 12px; animation-duration: 12s; animation-delay: 6s; }
.bubbles span:nth-child(4) { left: 42%; width: 24px; height: 24px; animation-duration: 17s; animation-delay: 1s; }
.bubbles span:nth-child(5) { left: 55%; width: 16px; height: 16px; animation-duration: 13s; animation-delay: 8s; }
.bubbles span:nth-child(6) { left: 68%; width: 34px; height: 34px; animation-duration: 21s; animation-delay: 4s; }
.bubbles span:nth-child(7) { left: 79%; width: 14px; height: 14px; animation-duration: 15s; animation-delay: 9s; }
.bubbles span:nth-child(8) { left: 90%; width: 26px; height: 26px; animation-duration: 18s; animation-delay: 2s; }
@keyframes rise {
    0%   { transform: translateY(0) translateX(0) scale(0.8); opacity: 0; }
    10%  { opacity: 0.9; }
    50%  { transform: translateY(-55vh) translateX(24px) scale(1); }
    100% { transform: translateY(-115vh) translateX(-18px) scale(1.1); opacity: 0; }
}
.block-container { position: relative; z-index: 1; }

/* keyframes */
@keyframes fadeUp { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn  { 0% { opacity: 0; transform: scale(0.85); } 70% { transform: scale(1.04); } 100% { opacity: 1; transform: scale(1); } }
@keyframes slideR { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
@keyframes shimmer { from { background-position: -200% 0; } to { background-position: 200% 0; } }
@keyframes pulseGlow { 0%,100% { box-shadow: 0 8px 18px rgba(255,107,87,0.35); } 50% { box-shadow: 0 8px 30px rgba(255,107,87,0.75); } }
@keyframes waveMove { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes heroFlow { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes swim { 0%,100% { transform: translateX(0) rotate(-6deg); } 50% { transform: translateX(-40px) rotate(6deg); } }

/* hero */
.hero-card { background-size: 200% 200%; animation: popIn .8s both, heroFlow 14s ease-in-out infinite .8s; }
.hero-card h1 { animation: slideR .9s .15s both; }
.hero-card p { animation: fadeUp .9s .3s both; }
.hero-kicker { animation: fadeUp .7s both; }
.step-pill { animation: fadeUp .7s both; }
.step-pill:nth-child(1) { animation-delay: .5s; } .step-pill:nth-child(2) { animation-delay: .62s; }
.step-pill:nth-child(3) { animation-delay: .74s; } .step-pill:nth-child(4) { animation-delay: .86s; }
.hero-card h1 span { background-size: 200% auto; background-image: linear-gradient(90deg, #ffd166, #ff8a7a, #7fd8e0, #ffd166); animation: shimmer 6s linear infinite; }

.hero-waves-wrap { position: absolute; left: 0; right: 0; bottom: 0; height: 70px; overflow: hidden; z-index: 1; }
.hero-waves-wrap svg { position: absolute; bottom: 0; left: 0; width: 200%; height: 70px; animation: waveMove 11s linear infinite; }
.hero-waves-wrap svg.slow { animation-duration: 19s; opacity: .55; }

.sea-deco { position: absolute; z-index: 1; opacity: .9; }
.sea-deco.fish1 { top: 18%; right: 26%; font-size: 1.6rem; animation: swim 9s ease-in-out infinite; }
.sea-deco.fish2 { bottom: 26%; right: 6%; font-size: 1.3rem; animation: swim 12s ease-in-out infinite reverse; }
.sea-deco.star  { bottom: 24%; left: 3%; font-size: 1.5rem; animation: floaty 6s ease-in-out infinite; }

/* stat tiles */
.stat-tile { animation: fadeUp .7s both; transition: transform .25s ease, box-shadow .25s ease; }
.stat-tile:nth-child(2) { animation-delay: .12s; } .stat-tile:nth-child(3) { animation-delay: .24s; }
.stat-tile:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 20px 36px rgba(11,79,108,.18); }
.stat-tile .ico { display: inline-block; transition: transform .4s ease; }
.stat-tile:hover .ico { transform: rotate(-12deg) scale(1.25); }

/* section headers */
.section-head { animation: slideR .6s both; }
.section-head .num { animation: pulseGlow 3s ease-in-out infinite; }
.section-sub { animation: fadeUp .7s .1s both; }
.section-head h2::after {
    content: ""; display: block; height: 4px; width: 56px; margin-top: 6px; border-radius: 4px;
    background: linear-gradient(90deg, var(--coral), var(--sun), var(--ocean-500), var(--coral));
    background-size: 200% 100%; animation: shimmer 4s linear infinite; transition: width .4s ease;
}
.section-head:hover h2::after { width: 120px; }

/* metrics */
div[data-testid="stMetric"] { position: relative; overflow: hidden; animation: popIn .6s both; }
div[data-testid="stMetric"]::after {
    content: ""; position: absolute; inset: 0; border-radius: 1.1rem; pointer-events: none;
    background: linear-gradient(115deg, transparent 40%, rgba(255,255,255,.65) 50%, transparent 60%);
    background-size: 250% 100%; background-position: 200% 0; transition: background-position .8s ease;
}
div[data-testid="stMetric"]:hover::after { background-position: -100% 0; }

.section-card { animation: fadeUp .7s both; transition: transform .3s ease, box-shadow .3s ease; }
.section-card:hover { transform: translateY(-3px); box-shadow: 0 20px 40px rgba(11,79,108,.16); }
.value-badge { transition: transform .2s ease, background .2s ease, color .2s ease; }
.value-badge:hover { transform: scale(1.06); background: var(--coral-soft); color: var(--coral); }

/* smooth interactive elements */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-testid="stSlider"] [role="slider"],
div[data-testid="stExpander"], div[data-testid="stAlert"], div[data-testid="stDataFrame"] {
    transition: all .25s cubic-bezier(.4,0,.2,1);
}
div[data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.35); }
div[data-testid="stExpander"]:hover, div[data-testid="stDataFrame"]:hover { transform: translateY(-2px); box-shadow: 0 16px 30px rgba(11,79,108,.14); }
div[data-testid="stAlert"] { animation: fadeUp .6s both; }
div[data-testid="stPlotlyChart"] { animation: fadeUp .8s both; border-radius: 1.2rem; overflow: hidden; }

div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background: linear-gradient(90deg, var(--ocean-500), var(--coral)) !important;
}

div[data-testid="stCheckbox"] { transition: transform .2s ease; }
div[data-testid="stCheckbox"]:hover { transform: translateX(4px); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--ocean-500), var(--coral)); border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }

.app-footer {
    background-size: 200% 100%;
    background-image: linear-gradient(90deg, rgba(15,163,177,.15), rgba(255,183,3,.15), rgba(255,107,87,.15), rgba(15,163,177,.15));
    animation: fadeUp .8s both, shimmer 10s linear infinite;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


def nice_model_name(stem: str) -> str:
    name = re.sub(r"[_-]+", " ", stem).strip()
    name = re.sub(r"\s+", " ", name).title()
    for old, new in (("Xgboost", "XGBoost"), ("Svr", "SVR"), ("Ann", "ANN")):
        name = name.replace(old, new)
    return name


def model_kind(path: Path) -> str:
    return "Keras neural network" if path.suffix.lower() in {".h5", ".keras"} else "Scikit-learn model"


def safe_key(name: str) -> str:
    return re.sub(r"\W+", "_", name).lower().strip("_")


def slider_settings(min_value: float, max_value: float) -> tuple[float, str]:
    span = max(max_value - min_value, 0.0)
    if span >= 10:
        return max(span / 100, 0.1), "%.1f"
    if span >= 1:
        return max(span / 100, 0.01), "%.2f"
    if span >= 0.1:
        return max(span / 100, 0.001), "%.3f"
    return max(span / 100, 0.00001), "%.5f"


def align_to_step(value: float, min_value: float, max_value: float, step: float) -> float:
    if step <= 0:
        return float(np.clip(value, min_value, max_value))
    aligned = min_value + round((value - min_value) / step) * step
    return float(np.clip(aligned, min_value, max_value))


def discover_model_files() -> list[dict[str, object]]:
    if not MODEL_DIR.exists():
        return []

    priority_map = {
        "random forest tuned": 0,
        "xgboost tuned": 1,
        "ann optimized": 2,
        "svr tuned": 3,
        "decision tree tuned": 4,
        "linear regression": 5,
    }

    entries: list[dict[str, object]] = []
    for path in MODEL_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
            continue
        label = nice_model_name(path.stem)
        entries.append(
            {
                "label": label,
                "path": path,
                "file_name": path.name,
                "kind": model_kind(path),
                "priority": priority_map.get(label.lower(), 100),
            }
        )

    return sorted(entries, key=lambda item: (item["priority"], str(item["label"]).lower()))


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH.name}")
        st.stop()

    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()

    if {"Temperature (Manuall)", "DO (Manual)"}.issubset(df.columns):
        df["Temp_DO"] = df["Temperature (Manuall)"] * df["DO (Manual)"]
    if {"pH (Manual)", "Ammonia (Manual)"}.issubset(df.columns):
        df["pH_Ammonia"] = df["pH (Manual)"] * df["Ammonia (Manual)"]

    return df


def target_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.strip().lower() in TARGET_NAMES]


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path_str: str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        return None, f"Model file not found: {model_path.name}"

    try:
        if model_path.suffix.lower() in {".h5", ".keras"}:
            from tensorflow.keras.models import load_model as load_keras_model

            return load_keras_model(model_path, compile=False), None

        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from sklearn.exceptions import InconsistentVersionWarning

                warnings.simplefilter("ignore", InconsistentVersionWarning)
            except Exception:
                pass

            try:
                return joblib.load(model_path), None
            except Exception:
                with model_path.open("rb") as handle:
                    return pickle.load(handle, encoding="latin1"), None
    except Exception as exc:
        return None, str(exc)


def get_model_feature_columns(model, df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(col) for col in getattr(model, "feature_names_in_")]
    return [col for col in df.columns if col not in target_cols]


def compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            stats[col] = {"min": 0.0, "max": 0.0, "median": 0.0}
        else:
            stats[col] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
    return stats


def render_parameter_selector(feature_cols: list[str]) -> list[str]:
    select_all = st.checkbox("Select all parameters", value=True, key="select_all_parameters")
    selected: list[str] = []
    selector_cols = st.columns(2)

    for index, col_name in enumerate(feature_cols):
        target_col = selector_cols[index % 2]
        with target_col:
            checked = st.checkbox(
                col_name,
                value=select_all,
                disabled=select_all,
                key=f"select_{safe_key(col_name)}",
            )
        if select_all or checked:
            selected.append(col_name)

    return feature_cols if select_all else selected


def render_exact_row_selector(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[bool, pd.DataFrame | None, dict[str, float] | None]:
    use_exact_row = st.checkbox(
        "Select an exact row from the dataset",
        value=False,
        key="use_exact_row_mode",
    )
    if not use_exact_row:
        return False, None, None

    st.info("When this option is enabled, the selected row values replace manual parameter entry.")

    preview_rows = df[feature_cols].copy()
    preview_rows.index = preview_rows.index + 1

    st.markdown("**Dataset preview**")
    st.dataframe(preview_rows.head(25), use_container_width=True, hide_index=False)

    row_options = list(preview_rows.index)
    selected_row_number = st.selectbox(
        "Choose one row",
        row_options,
        index=0,
        help="Select one record from the actual dataset. All parameters will be filled from this row.",
    )

    selected_row = preview_rows.loc[selected_row_number, feature_cols]
    selected_values = {col: float(selected_row[col]) for col in feature_cols}
    selected_frame = pd.DataFrame([selected_values])

    st.markdown("**Selected row values**")
    st.dataframe(selected_frame, use_container_width=True, hide_index=True)

    return True, selected_frame, selected_values


def render_parameters_section(
    df: pd.DataFrame,
    feature_stats: dict[str, dict[str, float]],
    feature_cols: list[str],
) -> tuple[dict[str, float], list[str], bool]:
    section_header(
        2,
        "🧪 Select parameters",
        "Choose the water-quality parameters you want to adjust, or pick one record from the dataset and use its values.",
    )

    use_exact_row, _, exact_row_values = render_exact_row_selector(df, feature_cols)
    if use_exact_row:
        selected_cols = feature_cols
        input_values = render_inputs(feature_stats, feature_cols, selected_cols, exact_row_values)
        return input_values, selected_cols, True

    selected_cols = render_parameter_selector(feature_cols)
    if not selected_cols:
        st.warning("Select at least one parameter to continue.")
        st.stop()

    input_values = render_inputs(feature_stats, feature_cols, selected_cols)
    return input_values, selected_cols, False


def render_inputs(
    feature_stats: dict[str, dict[str, float]],
    feature_cols: list[str],
    selected_cols: list[str],
    exact_row_values: dict[str, float] | None = None,
) -> dict[str, float]:
    section_header(
        3,
        "🎚️ Enter values",
        "Use the slider or type a value directly. Parameters that are not selected stay at their typical values.",
    )

    values: dict[str, float] = {}
    left_col, right_col = st.columns(2)
    for index, col_name in enumerate(feature_cols):
        stats = feature_stats[col_name]
        min_value = float(stats["min"])
        max_value = float(stats["max"])
        step, fmt = slider_settings(min_value, max_value)
        default_value = align_to_step(float(stats["median"]), min_value, max_value, step)
        slider_key = f"slider_{safe_key(col_name)}"
        number_key = f"number_{safe_key(col_name)}"
        target_col = left_col if index % 2 == 0 else right_col

        st.session_state.setdefault(slider_key, default_value)
        st.session_state.setdefault(number_key, default_value)

        def _sync_slider_to_number(s_key=slider_key, n_key=number_key):
            st.session_state[n_key] = st.session_state[s_key]

        def _sync_number_to_slider(n_key=number_key, s_key=slider_key):
            st.session_state[s_key] = st.session_state[n_key]

        with target_col:
            st.markdown(f"**{col_name}**")
            if exact_row_values is not None:
                exact_value = float(exact_row_values[col_name])
                st.markdown(f"<div class='value-badge'>Dataset row value: {exact_value:.5f}</div>", unsafe_allow_html=True)
                st.number_input(
                    "Row value",
                    min_value=min_value,
                    max_value=max_value,
                    value=exact_value,
                    step=step,
                    format=fmt,
                    key=f"row_{safe_key(col_name)}",
                    disabled=True,
                    label_visibility="collapsed",
                )
                values[col_name] = exact_value
                continue

            if col_name not in selected_cols:
                st.markdown(f"<div class='value-badge'>Typical value: {default_value:.5f}</div>", unsafe_allow_html=True)
                st.number_input(
                    "Typical value",
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=step,
                    format=fmt,
                    key=f"default_{safe_key(col_name)}",
                    disabled=True,
                    label_visibility="collapsed",
                )
                values[col_name] = default_value
                continue

            current_value = float(st.session_state[slider_key])
            st.markdown(f"<div class='value-badge'>Current value: {current_value:.5f}</div>", unsafe_allow_html=True)
            slider_col, number_col = st.columns([3, 1])
            with slider_col:
                st.slider(
                    "Slider",
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    format=fmt,
                    key=slider_key,
                    on_change=_sync_slider_to_number,
                    label_visibility="collapsed",
                )
            with number_col:
                st.number_input(
                    "Value",
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    format=fmt,
                    key=number_key,
                    on_change=_sync_number_to_slider,
                    label_visibility="collapsed",
                )

            values[col_name] = float(st.session_state[slider_key])

    return values


def add_engineered_features(input_df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    if "Temp_DO" in model_features and {"Temperature (Manuall)", "DO (Manual)"}.issubset(input_df.columns):
        input_df["Temp_DO"] = input_df["Temperature (Manuall)"] * input_df["DO (Manual)"]
    if "pH_Ammonia" in model_features and {"pH (Manual)", "Ammonia (Manual)"}.issubset(input_df.columns):
        input_df["pH_Ammonia"] = input_df["pH (Manual)"] * input_df["Ammonia (Manual)"]
    return input_df


def build_input_frame(input_values: dict[str, float], model_features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([input_values])
    frame = add_engineered_features(frame, model_features)
    for feature in model_features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    return frame.reindex(columns=model_features)


def normalize_prediction_array(prediction) -> np.ndarray:
    arr = np.asarray(prediction)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def primary_target_label(target_cols: list[str]) -> str:
    return next((c for c in target_cols if c.lower() == "weight"), target_cols[0] if target_cols else "Prediction")


def prediction_to_map(prediction, target_cols: list[str]) -> dict[str, float]:
    values = normalize_prediction_array(prediction)
    row = np.asarray(values[0]).reshape(-1)

    if row.size == 1:
        return {primary_target_label(target_cols): float(row[0])}

    mapped: dict[str, float] = {}
    for index, value in enumerate(row):
        label = target_cols[index] if index < len(target_cols) else f"Target {index + 1}"
        mapped[label] = float(value)
    return mapped


def prediction_value(prediction, target_cols: list[str]) -> float:
    arr = normalize_prediction_array(prediction)
    if arr.shape[1] == 1:
        return float(arr[0, 0])
    weight_index = next((i for i, col in enumerate(target_cols) if col.lower() == "weight"), 0)
    if weight_index < arr.shape[1]:
        return float(arr[0, weight_index])
    return float(arr[0, 0])


def predict_array(model, input_frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(input_frame))


def find_closest_history_row(
    df: pd.DataFrame, feature_cols: list[str], input_frame: pd.DataFrame
) -> tuple[pd.Series, float]:
    feature_matrix = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    query = input_frame.iloc[0][feature_cols].astype(float)

    center = feature_matrix.mean()
    spread = feature_matrix.std().replace(0, 1).fillna(1)
    normalized = (feature_matrix - center) / spread
    normalized_query = (query - center) / spread
    distances = np.sqrt(((normalized - normalized_query) ** 2).sum(axis=1))
    closest_index = int(distances.idxmin())
    return df.loc[closest_index], float(distances.loc[closest_index])


def percentage_difference(predicted_value: float, actual_value: float) -> tuple[float, float]:
    if np.isclose(actual_value, 0.0):
        return 0.0, 0.0
    signed = ((predicted_value - actual_value) / abs(actual_value)) * 100
    return float(signed), float(abs(signed))


def render_model_inventory(
    entries: list[dict[str, object]], loaded_models: dict[str, tuple[object | None, str | None]]
) -> None:
    rows = []
    for entry in entries:
        label = str(entry["label"])
        model, error = loaded_models.get(label, (None, "Not loaded"))
        rows.append(
            {
                "Model": label,
                "File": entry["file_name"],
                "Type": entry["kind"],
                "Status": "Ready" if model is not None and error is None else f"Needs attention: {error}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def compare_models(
    entries: list[dict[str, object]],
    loaded_models: dict[str, tuple[object | None, str | None]],
    input_frame: pd.DataFrame,
    actual_value: float,
    target_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for entry in entries:
        label = str(entry["label"])
        model, error = loaded_models.get(label, (None, None))
        if model is None:
            rows.append(
                {
                    "Model": label,
                    "Prediction": None,
                    "Actual": actual_value,
                    "Difference %": None,
                    "Status": f"Unavailable: {error}",
                }
            )
            continue

        try:
            predicted = prediction_value(predict_array(model, input_frame), target_cols)
            signed, absolute = percentage_difference(predicted, actual_value)
            rows.append(
                {
                    "Model": label,
                    "Prediction": round(predicted, 4),
                    "Actual": round(actual_value, 4),
                    "Difference %": round(absolute, 2),
                    "Signed %": round(signed, 2),
                    "Status": "Ready",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Model": label,
                    "Prediction": None,
                    "Actual": actual_value,
                    "Difference %": None,
                    "Status": f"Prediction failed: {exc}",
                }
            )

    return pd.DataFrame(rows)


def render_trend_chart(
    model,
    df: pd.DataFrame,
    feature_stats: dict[str, dict[str, float]],
    model_features: list[str],
    base_values: dict[str, float],
    trend_feature: str,
    weight_col: str,
    show_original: bool,
    chart_key: str,
) -> None:
    stats = feature_stats[trend_feature]
    min_value = float(stats["min"])
    max_value = float(stats["max"])
    trend_values = np.array([min_value]) if np.isclose(min_value, max_value) else np.linspace(min_value, max_value, 60)

    frame = pd.DataFrame([base_values] * len(trend_values))
    frame[trend_feature] = trend_values
    frame = add_engineered_features(frame, model_features)
    frame = frame.reindex(columns=model_features)
    predictions = predict_array(model, frame)
    predicted_curve = predictions.reshape(-1)

    current_frame = build_input_frame(base_values, model_features)
    current_prediction = prediction_value(predict_array(model, current_frame), target_columns(df))

    figure = go.Figure()
    if show_original:
        historical = df[[trend_feature, weight_col]].dropna().sort_values(trend_feature)
        figure.add_trace(
            go.Scatter(
                x=historical[trend_feature],
                y=historical[weight_col],
                mode="lines+markers",
                name="Historical data",
                line=dict(color="rgba(11, 79, 108, 0.35)", dash="dot"),
                marker=dict(size=4, color="rgba(11, 79, 108, 0.45)"),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=trend_values,
            y=predicted_curve,
            mode="lines",
            name="Model prediction",
            line=dict(color="#0fa3b1", width=4, shape="spline"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[base_values[trend_feature]],
            y=[current_prediction],
            mode="markers",
            name="Your current setting",
            marker=dict(size=15, color="#ff6b57", symbol="circle", line=dict(color="#ffffff", width=3)),
        )
    )

    figure.update_layout(
        margin=dict(t=50, l=20, r=20, b=20),
        height=420,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f3fafc",
        font=dict(color="#0b2a3c", family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=trend_feature,
        yaxis_title=weight_col.title(),
        xaxis=dict(gridcolor="rgba(11, 79, 108, 0.10)", zerolinecolor="rgba(11, 79, 108, 0.2)"),
        yaxis=dict(gridcolor="rgba(11, 79, 108, 0.10)", zerolinecolor="rgba(11, 79, 108, 0.2)"),
    )
    st.plotly_chart(figure, use_container_width=True, key=chart_key)


def main() -> None:
    st.markdown(
        """
        <div class="bubbles"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
        <div class="hero-card">
            <div class="hero-inner">
                <div class="hero-text">
                    <div class="hero-kicker">🌊 Smart aquaculture</div>
                    <h1>Shrimp Growth <span>Prediction Studio</span></h1>
                    <p>Pick a model, tweak your pond's water conditions, and instantly see how heavy your shrimp are expected to grow, checked against real farm records.</p>
                    <div class="step-strip">
                        <span class="step-pill">① Pick a model</span>
                        <span class="step-pill">② Select parameters</span>
                        <span class="step-pill">③ Enter values</span>
                        <span class="step-pill">④ Review the result</span>
                    </div>
                </div>
                <div class="hero-art">🦐</div>
            </div>
            <span class="sea-deco fish1">🐟</span>
            <span class="sea-deco fish2">🐠</span>
            <span class="sea-deco star">⭐</span>
            <div class="hero-waves-wrap">
                <svg class="slow" viewBox="0 0 2880 70" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                    <path fill="#f3fafc" d="M0,35 C240,80 480,0 720,30 C960,60 1200,10 1440,35 C1680,80 1920,0 2160,30 C2400,60 2640,10 2880,35 L2880,70 L0,70 Z"/>
                </svg>
                <svg viewBox="0 0 2880 70" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                    <path fill="#f3fafc" d="M0,50 C300,10 560,70 820,45 C1080,20 1280,60 1440,50 C1740,10 2000,70 2260,45 C2520,20 2720,60 2880,50 L2880,70 L0,70 Z"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-tile"><span class="ico">📊</span><div><div class="num">{len(df):,}</div><div class="lbl">Real farm records</div></div></div>
            <div class="stat-tile"><span class="ico">🧬</span><div><div class="num">{len(discover_model_files())}</div><div class="lbl">AI models ready</div></div></div>
            <div class="stat-tile"><span class="ico">💧</span><div><div class="num">{len(df.columns)}</div><div class="lbl">Data columns tracked</div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    target_cols = target_columns(df)
    if not target_cols:
        st.error("No target columns named Length or Weight were found in the dataset.")
        st.stop()

    model_entries = discover_model_files()
    if not model_entries:
        st.error(f"No model files were found in {MODEL_DIR.name}.")
        st.stop()

    loaded_models: dict[str, tuple[object | None, str | None]] = {}
    with st.spinner("Loading models..."):
        for entry in model_entries:
            label = str(entry["label"])
            loaded_models[label] = load_model_bundle(str(entry["path"]))

    available_entries = [entry for entry in model_entries if loaded_models.get(str(entry["label"]), (None, None))[0] is not None]
    if not available_entries:
        st.error("None of the models could be loaded.")
        st.stop()

    default_index = 0
    for index, entry in enumerate(available_entries):
        if "random forest" in str(entry["label"]).lower():
            default_index = index
            break

    section_header(1, "🤖 Choose a model", "Not sure? Keep the default, Random Forest, which is a reliable all-rounder.")
    model_label = st.selectbox(
        "Choose model",
        [str(entry["label"]) for entry in available_entries],
        index=default_index,
        help="All models are loaded from the model folder.",
    )

    selected_model, selected_error = loaded_models.get(model_label, (None, "Model not loaded"))
    if selected_model is None:
        st.error(f"{model_label} could not be loaded: {selected_error}")
        st.stop()

    model_features = get_model_feature_columns(selected_model, df, target_cols)
    base_feature_cols = [col for col in model_features if col not in ENGINEERED_FEATURES]
    missing_base = [col for col in base_feature_cols if col not in df.columns]
    if missing_base:
        st.error(f"Missing required data columns: {', '.join(missing_base)}")
        st.stop()

    numeric_cols = df[base_feature_cols].select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric input columns were found in the dataset.")
        st.stop()

    feature_stats = compute_feature_stats(df, numeric_cols)
    input_values, selected_cols, _ = render_parameters_section(df, feature_stats, numeric_cols)
    input_frame = build_input_frame(input_values, model_features)

    predicted_raw = predict_array(selected_model, input_frame)
    prediction_map = prediction_to_map(predicted_raw, target_cols)
    predicted_primary = prediction_value(predicted_raw, target_cols)

    closest_row, distance_score = find_closest_history_row(df, model_features, input_frame)
    primary_label = primary_target_label(target_cols)
    actual_primary = float(closest_row[primary_label])
    signed_pct, abs_pct = percentage_difference(predicted_primary, actual_primary)

    section_header(4, "🦐 Prediction result", "Here is what the model expects, compared with the most similar real record.")
    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        st.metric("Predicted weight", f"{predicted_primary:.3f}", delta=f"{signed_pct:+.1f}% versus the closest record")
    with top_mid:
        st.metric("Closest real record", f"{actual_primary:.3f}")
    with top_right:
        st.metric("Difference", f"{abs_pct:.1f}%")

    if abs_pct <= 10:
        st.success(
            f"The prediction is close to a real record in the dataset. It is {abs_pct:.1f}% {'higher' if signed_pct > 0 else 'lower'} than the nearest match."
        )
    elif abs_pct <= 20:
        st.info(
            f"The prediction is {abs_pct:.1f}% {'higher' if signed_pct > 0 else 'lower'} than the nearest historical record."
        )
    else:
        st.warning(
            f"The prediction is {abs_pct:.1f}% {'higher' if signed_pct > 0 else 'lower'} than the nearest historical record. Use it as an approximate reference."
        )

    show_detailed_analytics = st.checkbox("Show detailed analytics", value=False)
    show_live_graph = st.checkbox("Show live graph", value=True)
    live_graph_feature = None
    if show_live_graph:
        live_graph_feature = st.selectbox(
            "Live graph parameter",
            selected_cols,
            index=0,
            help="The graph updates as you change the selected parameter.",
        )

    result_cols = st.columns(min(len(prediction_map), 3))
    for index, (label, value) in enumerate(prediction_map.items()):
        result_cols[index % len(result_cols)].metric(label.title(), f"{value:.3f}")

    st.caption(
        f"Nearest historical match distance: {distance_score:.3f}. Use this with the percentage difference to assess model fit."
    )

    if show_live_graph and live_graph_feature is not None:
        section_header("📈", "Live graph", "This graph updates with the current input values and selected parameter.")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_trend_chart(
            selected_model,
            df,
            feature_stats,
            model_features,
            input_values,
            live_graph_feature,
            primary_label,
            True,
            "live_graph_chart",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if show_detailed_analytics:
        section_header("🔬", "Detailed analytics", "Compare every model side by side and explore trends.")
        comparison_df = compare_models(available_entries, loaded_models, input_frame, actual_primary, target_cols)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        with st.expander("Trend analysis", expanded=False):
            trend_feature = st.selectbox(
                "Choose a parameter to review",
                selected_cols,
                index=0,
                help="This shows how the selected model changes when one selected parameter changes.",
            )
            show_original = st.checkbox("Show historical data points", value=True)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_trend_chart(
                selected_model,
                df,
                feature_stats,
                model_features,
                input_values,
                trend_feature,
                primary_label,
                show_original,
                "detailed_trend_chart",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Available models", expanded=False):
            render_model_inventory(model_entries, loaded_models)

        with st.expander("Data preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

    st.markdown(
        "<div class='app-footer'>🦐 Shrimp Farm Helper · Predictions are estimates, so use them alongside your own farm judgement.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
