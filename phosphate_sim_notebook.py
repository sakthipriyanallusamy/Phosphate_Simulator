#pip install streamlit
streamlit run app.py

# %% [markdown]
# # Phosphate Breakthrough — AI Simulator (Notebook Style)
# Step-by-step cells you can run in VS Code (each `# %%` is a separate cell with a Run button).
# Make sure your CSV is at: `data/Phosphate_removal.csv` relative to where you run this file.

# %%
# 1) Imports & basic config
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

pd.set_option("display.max_columns", 120)
plt.rcParams["figure.figsize"] = (7, 4)
RNG = 42
print("Imports OK.")

# %%
# 2) Load data & basic cleaning
# Adjust the path if needed. Recommended structure:
# project/
#   data/Phosphate_removal.csv
#   phosphate_sim_notebook.py
data = pd.read_csv(r"D:\sp_python\Phosphate_Simulator\Phosphate_removal.csv")
data.columns = data.columns.str.strip()

print("Raw shape:", data.shape)
display(data.head(3))

# Ensure required columns
REQUIRED = [
    "Time_min","Ct/Co","Phosphate_adsorbed_%","Co_mg/L","Q_mL/min",
    "Diameter_cm","Length_cm","Porosity_mm","Iron_sludge_g","Sand_g",
    "Loading","Water_type"
]
missing = [c for c in REQUIRED if c not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Force numeric where appropriate
num_cols = ["Time_min","Ct/Co","Phosphate_adsorbed_%","Co_mg/L","Q_mL/min",
            "Diameter_cm","Length_cm","Porosity_mm","Iron_sludge_g","Sand_g"]
for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Encode categories
for cat in ["Loading","Water_type"]:
    if not np.issubdtype(data[cat].dtype, np.number):
        le = LabelEncoder()
        data[cat] = le.fit_transform(data[cat].astype(str))

# Drop rows with critical NaNs
data = data.dropna(subset=REQUIRED).reset_index(drop=True)
print("Cleaned shape:", data.shape)

# %%
# 3) Physics-informed feature engineering
area_cm2 = np.pi * (data["Diameter_cm"]/2.0)**2
Vb_cm3   = area_cm2 * data["Length_cm"]        # ~ mL
Q        = data["Q_mL/min"].replace(0, np.nan)

data["area_cm2"]       = area_cm2
data["Vb_cm3"]         = Vb_cm3.replace(0, np.nan)
data["EBCT_min"]       = data["Vb_cm3"] / Q
data["BV"]             = (Q * data["Time_min"]) / data["Vb_cm3"]
data["u_cm_min"]       = Q / data["area_cm2"]
data["Fe_g_per_mLbed"] = data["Iron_sludge_g"] / data["Vb_cm3"]

# Remove impossible rows from zero/NaN geometry
data = data.dropna(subset=["EBCT_min","BV","u_cm_min","Fe_g_per_mLbed"]).reset_index(drop=True)
print("Feature engineering done.")

# %%
# 4) Define features/targets
BASE_FEATURES = [
    "Time_min","Co_mg/L","Q_mL/min","Diameter_cm","Length_cm","Porosity_mm",
    "Iron_sludge_g","Sand_g","Loading","Water_type",
    "EBCT_min","BV","u_cm_min","Fe_g_per_mLbed",
]
y1_col = "Phosphate_adsorbed_%"
y2_col = "Ct/Co"

X  = data[BASE_FEATURES].copy()
y1 = data[y1_col].astype(float)
y2 = data[y2_col].astype(float)

print("Features ready:", len(BASE_FEATURES), "columns")

# %%
# 5) Split & train two Random Forest models
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=RNG)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=RNG)

rf1 = RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)  # Adsorbed %
rf2 = RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)  # Ct/Co

rf1.fit(X_train1, y_train1)
rf2.fit(X_train2, y_train2)

print("Training complete.")

# %%
# 6) Evaluate  — version-safe RMSE
import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("scikit-learn version:", sklearn.__version__)

def rmse_compat(y_true, y_pred):
    """Return RMSE; works on old and new scikit-learn versions."""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # older scikit-learn: no 'squared' kwarg
        return mean_squared_error(y_true, y_pred) ** 0.5

def eval_reg(y_true, y_pred, title=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = rmse_compat(y_true, y_pred)
    print(f"{title}\nR²={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}\n")

eval_reg(y_test1, rf1.predict(X_test1), "Model 1 — Phosphate Adsorbed (%)")
eval_reg(y_test2, rf2.predict(X_test2), "Model 2 — Ct/Co")


# %%
# 7) Diagnostics plots
def parity_plot(y_true, y_pred, title, xlab, ylab):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title); plt.grid(True); plt.show()

def residual_plot(y_true, y_pred, title):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title(title); plt.grid(True); plt.show()

def feature_importance_plot(model, feature_names, title):
    imp = model.feature_importances_
    order = np.argsort(imp)
    plt.figure()
    plt.barh(np.array(feature_names)[order], np.array(imp)[order])
    plt.title(title); plt.xlabel("Importance"); plt.tight_layout(); plt.grid(True, axis="x", alpha=0.3); plt.show()

# Model 1
y1_pred = rf1.predict(X_test1)
parity_plot(y_test1, y1_pred, "Parity — Phosphate Adsorbed (%)", "Actual", "Predicted")
residual_plot(y_test1, y1_pred, "Residuals — Phosphate Adsorbed")
feature_importance_plot(rf1, X_train1.columns, "Importance — Model 1 (Adsorbed %)")

# Model 2
y2_pred = rf2.predict(X_test2)
parity_plot(y_test2, y2_pred, "Parity — Ct/Co", "Actual", "Predicted")
residual_plot(y_test2, y2_pred, "Residuals — Ct/Co")
feature_importance_plot(rf2, X_train2.columns, "Importance — Model 2 (Ct/Co)")

# %%
# 8) (Optional) SHAP global explanations for rf1
# If not installed: pip install shap
try:
    import shap
    shap_explainer = shap.TreeExplainer(rf1)
    shap_vals = shap_explainer.shap_values(X_train1)
    shap.summary_plot(shap_vals, X_train1, plot_type="bar")
    shap.summary_plot(shap_vals, X_train1)
except Exception as e:
    print("SHAP skipped or failed:", e)

# %%
# 9) Helpers for breakthrough simulation
FEATURES_USED = list(X_train2.columns)   # exact order used to fit rf2
TIME_COL = "Time_min"

def _is_int_series(s: pd.Series) -> bool:
    try: return np.array_equal(s.dropna(), s.dropna().astype(int))
    except Exception: return False

def make_baseline_row(df: pd.DataFrame, features):
    base = {}
    for c in features:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            base[c] = (float(s.mode().iloc[0]) if s.nunique()<=5 or _is_int_series(s)
                       else float(s.median()))
        else:
            base[c] = s.mode().iloc[0]
    return pd.Series(base, index=features)

def build_grid(df, features, fixed, t_array):
    row = make_baseline_row(df, features).copy()
    for k,v in (fixed or {}).items():
        if k in row.index: row[k] = v
    Xg = pd.DataFrame([row.values]*len(t_array), columns=row.index)
    if TIME_COL not in Xg.columns:
        raise KeyError(f"{TIME_COL} not in training features; retrain including time.")
    Xg[TIME_COL] = t_array
    return Xg[features]

print("Simulator helpers ready.")

# %%
# 10) Single-condition predicted breakthrough curve
tmin, tmax = float(data[TIME_COL].min()), float(data[TIME_COL].max())
t = np.linspace(tmin, tmax, 200)

fixed = {
    "Water_type": int(data["Water_type"].mode().iloc[0]),
    "Loading":    int(data["Loading"].mode().iloc[0]),
    "Co_mg/L":    float(data["Co_mg/L"].median()),
    "Q_mL/min":   float(data["Q_mL/min"].median()),
    "Sand_g":     float(data["Sand_g"].median()),
    "Iron_sludge_g": float(data["Iron_sludge_g"].median()),
    "Diameter_cm": float(data["Diameter_cm"].median()),
    "Length_cm":   float(data["Length_cm"].median()),
    "Porosity_mm": float(data["Porosity_mm"].median()),
}
Xg = build_grid(data, FEATURES_USED, fixed, t)
yhat = rf2.predict(Xg)

plt.figure()
plt.plot(t, yhat)
plt.xlabel("Time (min)"); plt.ylabel("Ct/Co"); plt.title("Predicted Breakthrough (single condition)")
plt.grid(True); plt.tight_layout(); plt.show()

# %%
# 11) Compare curves by Water Type
plt.figure()
for wt in sorted(data["Water_type"].unique()):
    fx = dict(fixed); fx["Water_type"] = int(wt)
    Xg = build_grid(data, FEATURES_USED, fx, t)
    yhat = rf2.predict(Xg)
    plt.plot(t, yhat, label=f"Water_type={wt}")
plt.xlabel("Time (min)"); plt.ylabel("Ct/Co"); plt.title("Comparison by Water Type")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# %%
# 12) Interactive simulator (ipywidgets)
# Works in VS Code Python Interactive or Jupyter; requires ipywidgets installed.
try:
    from ipywidgets import interact, FloatSlider, IntSlider, Play, jslink, HBox, VBox

    def nice_range(col, step=None):
        s = data[col]; vmin, vmax = float(s.min()), float(s.max()); vmed=float(s.median())
        if step is None: step = max((vmax-vmin)/100.0, 1e-4)
        return vmin, vmax, vmed, step

    controls = {}
    for c in FEATURES_USED:
        if c == TIME_COL: continue
        vmin,vmax,vmed,step = nice_range(c)
        if "Q_mL/min" in c: step = 0.5
        if "Co_mg/L"  in c: step = 0.1
        if c.endswith(("cm","mm")): step = 0.1
        if "Sand_g" in c or "Iron_sludge_g" in c: step = 1.0
        s = data[c]
        if _is_int_series(s) and s.nunique() <= 12:
            controls[c] = IntSlider(min=int(vmin), max=int(vmax), value=int(round(vmed)), step=1, description=c)
        else:
            controls[c] = FloatSlider(min=vmin, max=vmax, value=vmed, step=step, description=c)

    tmin, tmax = float(data[TIME_COL].min()), float(data[TIME_COL].max())
    play = Play(value=int(tmin), min=int(tmin), max=int(tmax), step=1, interval=120, description="▶")
    time_slider = IntSlider(min=int(tmin), max=int(tmax), step=1, value=int(tmin), description="Time (min)")
    jslink((play, 'value'), (time_slider, 'value'))

    def live_curve(Time_min, **kwargs):
        T = np.linspace(tmin, Time_min, max(2, int(Time_min - tmin + 2)))
        fixed_now = {k: v for k,v in kwargs.items() if k in FEATURES_USED}
        Xg = build_grid(data, FEATURES_USED, fixed_now, T)
        yhat = rf2.predict(Xg)
        plt.figure(figsize=(6,4))
        plt.plot(T, yhat)
        plt.xlabel("Time (min)"); plt.ylabel("Ct/Co")
        plt.title("Real-time Ct/Co prediction"); plt.grid(True); plt.tight_layout(); plt.show()

    # Layout
    rows, items = [], list(controls.items())
    for i in range(0, len(items), 2):
        rows.append(HBox([w for _,w in items[i:i+2]]))
    ui = VBox([HBox([play, time_slider])] + rows)

    kwargs = {"Time_min": time_slider}; kwargs.update(controls)
    interact(live_curve, **kwargs)
    display(ui)
except Exception as e:
    print("ipywidgets simulator skipped or failed:", e)

# %%
# 13) Export current curve to CSV (create folder if needed)
import os
os.makedirs("models", exist_ok=True)

T = np.linspace(tmin, tmax, 200)
Xg = build_grid(data, FEATURES_USED, fixed, T)
yhat = rf2.predict(Xg)
out_df = pd.DataFrame({"Time_min": T, "Pred_CtCo": yhat})
out_path = "models/predicted_breakthrough_export.csv"
out_df.to_csv(out_path, index=False)
print("Saved:", out_path)

# %%
# 14) Save artifacts for Streamlit reuse
import pickle
with open("models/rf2.pkl","wb") as f: pickle.dump(rf2, f)
with open("models/features.pkl","wb") as f: pickle.dump(FEATURES_USED, f)
data.to_csv("models/data_for_ranges.csv", index=False)
print("Saved: models/rf2.pkl, models/features.pkl, models/data_for_ranges.csv")

# %% [markdown]
# 15) Streamlit app (run separately in a terminal)
# Save/keep the generated app.py (provided alongside this file) and execute:
#     pip install streamlit
#     streamlit run app.py

# %% Train and save the adsorption model (rf1)

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the same data you used for rf2 ranges
data = pd.read_csv("models/data_for_ranges.csv")

# Use the exact same feature order used for rf2
FEATURES_USED = pickle.load(open("models/features.pkl","rb"))

# Target for adsorption (must exist in your data)
TARGET_ADS = "Phosphate_adsorbed_%"
if TARGET_ADS not in data.columns:
    raise ValueError(f"Column '{TARGET_ADS}' not found in data_for_ranges.csv")

X = data[FEATURES_USED].copy()
y = data[TARGET_ADS].astype(float)

# Basic split + model
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rf1 = RandomForestRegressor(n_estimators=300, random_state=42)
rf1.fit(Xtr, ytr)

# Optional quick metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
yhat = rf1.predict(Xte)
print(f"Adsorption model — R2={r2_score(yte,yhat):.3f}, MAE={mean_absolute_error(yte,yhat):.3f}, RMSE={mean_squared_error(yte,yhat)**0.5:.3f}")

# Save the adsorption model
import os
os.makedirs("models", exist_ok=True)
with open("models/rf1.pkl","wb") as f:
    pickle.dump(rf1, f)

print("Saved models/rf1.pkl")

# %%
