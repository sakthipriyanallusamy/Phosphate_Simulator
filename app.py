# app.py  (COPY–PASTE THIS FULL FILE)
# Streamlit Phosphate Column Simulator — robust for Cloud (no series.min() crash)

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==============================
# Constants / Paths
# ==============================
TIME_COL   = "Time_min"
MODEL_CTCO = "models/rf2.pkl"              # Ct/Co model
MODEL_ADS  = "models/rf1.pkl"              # Adsorption model
FEATS_PATH = "models/features.pkl"         # feature list fallback
DATA_PATH  = "models/data_for_ranges.csv"  # dataset for slider ranges

NA_STRINGS = {"Not_applicable", "NA", "N/A", "NaN", "None", "none", "-", ""}

# ==============================
# Utility: load pickle safely
# ==============================
def safe_load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

def get_feature_list(rf2, rf1):
    if rf2 is not None and hasattr(rf2, "feature_names_in_"):
        return list(rf2.feature_names_in_)
    if rf1 is not None and hasattr(rf1, "feature_names_in_"):
        return list(rf1.feature_names_in_)
    if os.path.exists(FEATS_PATH):
        return list(safe_load_pickle(FEATS_PATH))
    st.error("Cannot determine feature list. Provide models/features.pkl or retrain with feature_names_in_.")
    st.stop()

# ==============================
# Utility: data preparation
# ==============================
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(list(NA_STRINGS), np.nan)
    return df

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().sum() == 0:
                df[c] = 0.0
    return df

def encode_categories_to_codes(df: pd.DataFrame, cat_cols):
    """
    Returns:
      original_series_dict: original (labels) for display
      df_codes: df where cat cols are replaced with integer codes (float)
    """
    original = {}
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            original[c] = out[c].copy()
            out[c] = out[c].astype("category")
            out[c] = out[c].cat.codes.astype(float)   # model-friendly
            out[c] = out[c].replace(-1, 0.0)
    return original, out

# ==============================
# Engineered features (safe)
# ==============================
def add_engineered_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    # Make sure we always have Series (not scalars) and numeric
    D  = pd.to_numeric(out.get("Diameter_cm",    pd.Series([0]*n)), errors="coerce").fillna(0.0)
    L  = pd.to_numeric(out.get("Length_cm",      pd.Series([0]*n)), errors="coerce").fillna(0.0)
    Q  = pd.to_numeric(out.get("Q_mL/min",       pd.Series([0]*n)), errors="coerce").fillna(0.0)
    Fe = pd.to_numeric(out.get("Iron_sludge_g",  pd.Series([0]*n)), errors="coerce").fillna(0.0)
    t  = pd.to_numeric(out.get(TIME_COL,         pd.Series([0]*n)), errors="coerce").fillna(0.0)

    area_cm2 = np.pi * (D / 2.0) ** 2
    Vb_cm3   = area_cm2 * L

    # Avoid divide-by-zero
    Qnz  = Q.replace(0, np.nan)
    Vbnz = Vb_cm3.replace(0, np.nan)
    Anz  = area_cm2.replace(0, np.nan)

    out["EBCT_min"]       = (Vb_cm3 / Qnz)
    out["BV"]             = (Q * t) / Vbnz
    out["u_cm_min"]       = (Q / Anz)
    out["Fe_g_per_mLbed"] = (Fe / Vbnz)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

# ==============================
# Baseline + Grid builder
# ==============================
def make_baseline_row(df: pd.DataFrame, features):
    row = {}
    for c in features:
        if c not in df.columns:
            row[c] = 0.0
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce")
            s = s[np.isfinite(s)]
            row[c] = float(s.median()) if not s.empty else 0.0
        else:
            m = df[c].mode(dropna=True)
            row[c] = m.iloc[0] if len(m) else 0
    return pd.Series(row, index=features)

def build_grid(df: pd.DataFrame, features, fixed: dict, t_array: np.ndarray):
    base = make_baseline_row(df, features).copy()
    for k, v in fixed.items():
        if k in base.index:
            base[k] = v

    X = pd.DataFrame([base.values] * len(t_array), columns=features)

    if TIME_COL not in X.columns:
        st.error(f"'{TIME_COL}' is not in FEATURES_USED. Retrain including {TIME_COL}.")
        st.stop()

    X[TIME_COL] = t_array

    # add engineered columns
    X = add_engineered_cols(X)

    # Ensure all required columns exist
    for c in features:
        if c not in X.columns:
            X[c] = 0.0

    return X[features]

# ==============================
# Slider helpers (NO series.min() on raw strings)
# ==============================
def numeric_slider(name: str, series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]

    if s.empty:
        st.sidebar.write(f"{name}: **n/a**")
        return 0.0

    vmin, vmax, vmed = float(s.min()), float(s.max()), float(s.median())

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        st.sidebar.write(f"{name}: **{vmed:.3g}** (fixed)")
        return float(vmed)

    step = max((vmax - vmin) / 100.0, 1e-6)
    return float(st.sidebar.slider(name, vmin, vmax, vmed, step=step))

def categorical_select_as_code(name: str, original_series: pd.Series):
    s = original_series.astype("category")
    labels = list(s.cat.categories)

    if len(labels) == 0:
        st.sidebar.write(f"{name}: **n/a**")
        return 0

    idx = st.sidebar.selectbox(name, list(range(len(labels))), format_func=lambda i: str(labels[i]), index=0)
    return int(idx)

# ==============================
# Predict safely
# ==============================
def predict_safe(model, X: pd.DataFrame):
    try:
        return model.predict(X)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("X columns:", list(X.columns))
        st.dataframe(X.head())
        st.stop()

# ==============================
# App start
# ==============================
st.set_page_config(page_title="Phosphate Column Simulator", layout="wide")
st.title("💧 Phosphate Column Simulator")
st.caption("Build: 2025-12-29 (robust sliders + safe engineered features)")

# Required files
missing = [p for p in [DATA_PATH] if not os.path.exists(p)]
if missing:
    st.error(f"Missing files: {missing}. Make sure they exist in your repo.")
    st.stop()

# Load models (one or both)
rf2 = safe_load_pickle(MODEL_CTCO) if os.path.exists(MODEL_CTCO) else None
rf1 = safe_load_pickle(MODEL_ADS)  if os.path.exists(MODEL_ADS)  else None

if rf2 is None and rf1 is None:
    st.error("No models found. Please include models/rf1.pkl and/or models/rf2.pkl.")
    st.stop()

# Load data + features
data = prepare_data(pd.read_csv(DATA_PATH))
FEATURES_USED = get_feature_list(rf2, rf1)

if TIME_COL not in data.columns:
    st.error(f"'{TIME_COL}' column not found in {DATA_PATH}.")
    st.stop()

# Make time numeric early
data[TIME_COL] = pd.to_numeric(data[TIME_COL], errors="coerce").fillna(0.0)

# Categorical columns you want as selectbox
CAT_COLS = [c for c in ["Loading", "Water_type"] if (c in FEATURES_USED and c in data.columns)]
NUM_COLS = [c for c in FEATURES_USED if c not in CAT_COLS]

data = ensure_numeric(data, NUM_COLS)
cat_original, data_codes = encode_categories_to_codes(data, CAT_COLS)

# ==============================
# Sidebar inputs
# ==============================
st.sidebar.header("Fixed Conditions")
fixed = {}

for c in FEATURES_USED:
    if c == TIME_COL:
        continue
    if c in CAT_COLS:
        fixed[c] = categorical_select_as_code(c, cat_original[c])
    else:
        if c not in data_codes.columns:
            fixed[c] = 0.0
        else:
            fixed[c] = numeric_slider(c, data_codes[c])

# Time range + points
t_series = pd.to_numeric(data_codes[TIME_COL], errors="coerce")
t_series = t_series[np.isfinite(t_series)]
tmin, tmax = float(t_series.min()), float(t_series.max())

if tmin == tmax:
    st.sidebar.write(f"Time range: **{tmin} min** (fixed)")
    t_start, t_end = tmin, tmax
else:
    t_start, t_end = st.sidebar.slider("Time range (min)", tmin, tmax, (tmin, tmax))

n_pts = st.sidebar.slider("Points", 50, 500, 200)

# Build grid
t = np.linspace(t_start, t_end, n_pts)
Xg = build_grid(data_codes, FEATURES_USED, fixed, t)

# ==============================
# Output tabs
# ==============================
tab_labels = []
if rf2 is not None:
    tab_labels.append("Breakthrough (Ct/Co)")
if rf1 is not None:
    tab_labels.append("Phosphate adsorption (%)")

tabs = st.tabs(tab_labels)

idx = 0

# Ct/Co
if rf2 is not None:
    with tabs[idx]:
        y_ctco = predict_safe(rf2, Xg)
        fig, ax = plt.subplots()
        ax.plot(t, y_ctco)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Ct/Co")
        ax.set_title("Predicted Breakthrough Curve")
        ax.grid(True)
        st.pyplot(fig)

        out = pd.DataFrame({"Time_min": t, "Pred_CtCo": y_ctco})
        st.download_button("Download Ct/Co CSV", out.to_csv(index=False), "predicted_breakthrough_ctco.csv", "text/csv")
    idx += 1

# Adsorption
if rf1 is not None:
    with tabs[idx if rf2 is not None else 0]:
        y_ads = predict_safe(rf1, Xg)
        fig2, ax2 = plt.subplots()
        ax2.plot(t, y_ads)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Phosphate adsorbed (%)")
        ax2.set_title("Predicted Phosphate Adsorption")
        ax2.grid(True)
        st.pyplot(fig2)

        out2 = pd.DataFrame({"Time_min": t, "Pred_Adsorption_pct": y_ads})
        st.download_button("Download Adsorption CSV", out2.to_csv(index=False), "predicted_adsorption_pct.csv", "text/csv")

# Debug panel
with st.expander("Debug (inputs and model matrix)"):
    st.write("Fixed values sent to model:", fixed)
    st.write("FEATURES_USED:", FEATURES_USED)
    st.dataframe(Xg.head(10))
