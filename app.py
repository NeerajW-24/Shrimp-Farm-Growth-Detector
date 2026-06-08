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
html, body, [data-testid="stAppViewContainer"] {
    background: #070b14;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #05070d 100%);
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}

.hero-card {
    padding: 1.15rem 1.2rem;
    border-radius: 1.25rem;
    border: 1px solid rgba(129, 140, 248, 0.25);
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f1147 100%);
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.35);
}

.hero-kicker {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #f87171;
    margin-bottom: 0.35rem;
}

.hero-card h1 {
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.05;
    margin: 0 0 0.55rem 0;
    color: #f8fafc;
}

.hero-card p {
    margin: 0;
    font-size: 1rem;
    color: #e2e8f0;
}

.step-strip {
    margin-top: 0.9rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
}

.step-pill {
    padding: 0.42rem 0.7rem;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.18);
    color: #f8fafc;
    font-size: 0.9rem;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
    border: 1px solid rgba(96, 165, 250, 0.16);
    border-radius: 1rem;
    padding: 0.95rem 1rem;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
}

.small-note {
    color: #cbd5e1;
    font-size: 0.92rem;
}

.value-badge {
    margin: 0.2rem 0 0.55rem 0;
    display: inline-block;
    padding: 0.28rem 0.58rem;
    border-radius: 999px;
    background: rgba(147, 51, 234, 0.18);
    color: #f8fafc;
    border: 1px solid rgba(244, 114, 182, 0.22);
    font-size: 0.84rem;
    font-weight: 700;
}

section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #f8fafc !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="slider"] {
    background: #0b1020 !important;
    color: #f8fafc !important;
    border-color: rgba(96, 165, 250, 0.35) !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background: #ef4444 !important;
    box-shadow: 0 0 0 0.25rem rgba(147, 51, 234, 0.18) !important;
}

.section-card {
    padding: 1rem 1.1rem;
    border-radius: 1rem;
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border: 1px solid rgba(96, 165, 250, 0.14);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
}

.section-card h3, .section-card p, .section-card label {
    color: #f8fafc !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 1rem;
    overflow: hidden;
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
    st.subheader("2. Select parameters")
    st.caption(
        "Choose the parameters you want to adjust, or select one record from the dataset and use its values."
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
    st.subheader("3. Enter values")
    st.caption("Use the slider or type a value directly. Parameters that are not selected stay at their typical values.")

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
                line=dict(color="rgba(90, 110, 100, 0.5)", dash="dot"),
                marker=dict(size=4, color="rgba(90, 110, 100, 0.45)"),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=trend_values,
            y=predicted_curve,
            mode="lines",
            name="Model prediction",
            line=dict(color="#1b6b4b", width=4),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[base_values[trend_feature]],
            y=[current_prediction],
            mode="markers",
            name="Your current setting",
            marker=dict(size=12, color="#e85d04", symbol="circle"),
        )
    )

    figure.update_layout(
        margin=dict(t=50, l=20, r=20, b=20),
        height=420,
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=trend_feature,
        yaxis_title=weight_col.title(),
        xaxis=dict(
            color="#f8fafc",
            gridcolor="rgba(148, 163, 184, 0.18)",
            zerolinecolor="rgba(248, 250, 252, 0.25)",
            tickfont=dict(color="#f8fafc"),
        ),
        yaxis=dict(
            color="#f8fafc",
            gridcolor="rgba(148, 163, 184, 0.18)",
            zerolinecolor="rgba(248, 250, 252, 0.25)",
            tickfont=dict(color="#f8fafc"),
        ),
    )
    st.plotly_chart(figure, use_container_width=True, key=chart_key)


def main() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Shrimp growth analysis</div>
            <h1>Prediction dashboard</h1>
            <p>Select a model, choose the parameters you want to adjust, and review the prediction against the closest record in the dataset.</p>
            <div class="step-strip">
                <span class="step-pill">1. Pick a model</span>
                <span class="step-pill">2. Select parameters</span>
                <span class="step-pill">3. Review the result</span>
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

    st.markdown("### 1. Choose a model")
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

    st.markdown("### 4. Prediction result")
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
        st.markdown("### Live graph")
        st.caption("This graph updates with the current input values and selected parameter.")
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
        st.markdown("### Detailed analytics")
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
            st.markdown('<div class="hero-card">', unsafe_allow_html=True)
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


if __name__ == "__main__":
    main()
