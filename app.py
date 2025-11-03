import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==============================
# Paths & constants
# ==============================
TIME_COL   = "Time_min"
MODEL_CTCO = "models/rf2.pkl"              # Ct/Co model
MODEL_ADS  = "models/rf1.pkl"              # Phosphate_adsorbed_% model (optional)
FEATS_PATH = "models/features.pkl"         # exact feature order used in training
DATA_PATH  = "models/data_for_ranges.csv"  # dataset for slider ranges

# ==============================
# Guards & loads
# ==============================
missing = [p for p in (FEATS_PATH, DATA_PATH) if not os.path.exists(p)]
if missing:
    st.error(f"Missing files: {missing}. Run your training/notebook to create artifacts in 'models/'.")
    st.stop()

FEATURES_USED = pickle.load(open(FEATS_PATH, "rb"))
data = pd.read_csv(DATA_PATH)

rf2 = pickle.load(open(MODEL_CTCO, "rb")) if os.path.exists(MODEL_CTCO) else None
rf1 = pickle.load(open(MODEL_ADS,  "rb")) if os.path.exists(MODEL_ADS)  else None
if rf2 is None and rf1 is None:
    st.error("No models found. Need models/rf2.pkl and/or models/rf1.pkl.")
    st.stop()

# ==============================
# Helpers
# ==============================
def _is_int_series(s: pd.Series) -> bool:
    try:
        return np.array_equal(s.dropna(), s.dropna().astype(int))
    except Exception:
        return False

def make_baseline_row(df: pd.DataFrame, features):
    base = {}
    for c in features:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            base[c] = float(s.median()) if s.nunique() > 5 else float(s.mode(dropna=True).iloc[0])
        else:
            base[c] = s.mode(dropna=True).iloc[0]
    return pd.Series(base, index=features)

def build_grid(df: pd.DataFrame, features, fixed: dict, t_array: np.ndarray):
    row = make_baseline_row(df, features).copy()
    for k, v in (fixed or {}).items():
        if k in row.index:
            row[k] = v
    Xg = pd.DataFrame([row.values] * len(t_array), columns=row.index)
    if TIME_COL not in Xg.columns:
        st.error(f"'{TIME_COL}' not in FEATURES_USED. Retrain including time or adjust the app.")
        st.stop()
    Xg[TIME_COL] = t_array
    return Xg[features]

def make_slider(label: str, series: pd.Series, prefer_int: bool = False):
    vmin = float(series.min())
    vmax = float(series.max())
    vmed = float(series.median())

    # If column has no variation: show fixed value (no slider)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        st.sidebar.write(f"{label}: **{int(round(vmed)) if prefer_int else vmed}** (fixed)")
        return int(round(vmed)) if prefer_int else vmed

    # Step size
    step = max((vmax - vmin) / 100.0, 1e-6)

    if prefer_int:
        return st.sidebar.slider(
            label,
            int(np.floor(vmin)),
            int(np.ceil(vmax)),
            int(round(vmed)),
            step=1,
        )
    else:
        # Niceties for certain fields
        if "Q_mL/min" in label:
            step = max(step, 0.5)
        if "Co_mg/L" in label:
            step = max(step, 0.1)
        if label.endswith(("cm", "mm")):
            step = max(step, 0.1)
        if "Sand_g" in label or "Iron_sludge_g" in label:
            step = max(step, 1.0)
        return st.sidebar.slider(label, vmin, vmax, vmed, step=step)

# ==============================
# UI
# ==============================
st.title("Phosphate Column Simulator")

st.sidebar.header("Fixed Conditions")
fixed = {}
for c in FEATURES_USED:
    if c == TIME_COL:
        continue
    s = data[c]
    prefer_int = (_is_int_series(s) and s.nunique() <= 12)
    fixed[c] = make_slider(c, s, prefer_int=prefer_int)

# Time controls
tmin = float(data[TIME_COL].min())
tmax = float(data[TIME_COL].max())
if tmin == tmax:
    st.sidebar.write(f"Time range: **{tmin} min** (fixed)")
    t_start, t_end = tmin, tmax
else:
    t_start, t_end = st.sidebar.slider("Time range (min)", tmin, tmax, (tmin, tmax))

n_pts = st.sidebar.slider("Points", 50, 500, 200)

# Build grid and predict
t = np.linspace(t_start, t_end, n_pts)
Xg = build_grid(data, FEATURES_USED, fixed, t)

# Tabs (Ct/Co + Adsorption)
tabs = []
labels = []
if rf2 is not None:
    labels.append("Breakthrough (Ct/Co)")
    tabs.append("ctco")
if rf1 is not None:
    labels.append("Phosphate adsorption (%)")
    tabs.append("ads")

stabs = st.tabs(labels)
i = 0

if rf2 is not None:
    with stabs[i]:
        y_ctco = rf2.predict(Xg)
        fig, ax = plt.subplots()
        ax.plot(t, y_ctco)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Ct/Co")
        ax.set_title("Predicted Breakthrough (Ct/Co vs Time)")
        ax.grid(True)
        st.pyplot(fig)

        if st.button("Export Ct/Co CSV"):
            os.makedirs("models", exist_ok=True)
            pd.DataFrame({"Time_min": t, "Pred_CtCo": y_ctco}).to_csv(
                "models/predicted_breakthrough_ctco.csv", index=False
            )
            st.success("Saved: models/predicted_breakthrough_ctco.csv")
    i += 1

if rf1 is not None:
    with stabs[i if rf2 is not None else 0]:
        y_ads = rf1.predict(Xg)
        fig2, ax2 = plt.subplots()
        ax2.plot(t, y_ads)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Phosphate adsorbed (%)")
        ax2.set_title("Predicted Phosphate Adsorption (% vs Time)")
        ax2.grid(True)
        st.pyplot(fig2)

        if st.button("Export adsorption CSV"):
            os.makedirs("models", exist_ok=True)
            pd.DataFrame({"Time_min": t, "Pred_Adsorption_pct": y_ads}).to_csv(
                "models/predicted_adsorption_pct.csv", index=False
            )
            st.success("Saved: models/predicted_adsorption_pct.csv")
