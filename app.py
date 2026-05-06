import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #f8fafc !important; }

    .main { background-color: #f8fafc; }

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .metric-card.danger  { border-left-color: #ef4444; }
    .metric-card.success { border-left-color: #22c55e; }
    .metric-card.warning { border-left-color: #f59e0b; }

    .result-positive {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 1px solid #fca5a5;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #86efac;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    label { font-weight: 500 !important; color: #374151 !important; }

    .section-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(59,130,246,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        box-shadow: 0 4px 16px rgba(59,130,246,0.45);
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("preprocessing_artifacts_xgb.pkl", "rb") as f:
        arts = pickle.load(f)
    with open("xgb_platt_model.pkl", "rb") as f:
        model = pickle.load(f)
    scaler            = arts["scaler"]
    selected_features = arts["selected_features"]
    feature_order     = arts["feature_order"]
    iqr_bounds        = arts["iqr_bounds"]
    return scaler, selected_features, feature_order, iqr_bounds, model

scaler, selected_features, feature_order, iqr_bounds, model = load_artifacts()
train_means = {feat: scaler.mean_[i] for i, feat in enumerate(feature_order)}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Troponin_log"]   = np.log1p(df["Troponin"])
    df["CKMB_log"]       = np.log1p(df["CK-MB"])
    df["BloodSugar_log"] = np.log1p(df["Blood sugar"])
    df["pulse_pressure"] = df["Systolic blood pressure"] - df["Diastolic blood pressure"]
    df["map"]            = (df["Systolic blood pressure"] + 2 * df["Diastolic blood pressure"]) / 3
    return df


def apply_iqr_capping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, bounds in iqr_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])
    return df


def align_and_impute(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_eng = engineer_features(df_raw)
    df_cap = apply_iqr_capping(df_eng)
    for col in feature_order:
        if col not in df_cap.columns:
            df_cap[col] = np.nan
    X = df_cap.reindex(columns=feature_order)
    X = X.apply(pd.to_numeric, errors="coerce")
    for col in feature_order:
        X[col] = X[col].fillna(train_means.get(col, 0.0))
    return X


def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    X           = align_and_impute(df_raw)
    X_scaled    = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order, index=df_raw.index)
    return X_scaled_df[selected_features]


def predict_df(df_raw: pd.DataFrame):
    X_rfe = preprocess_df(df_raw)
    proba = model.predict_proba(X_rfe)
    if hasattr(model, "classes_") and 1 in list(model.classes_):
        pos_idx = list(model.classes_).index(1)
    else:
        pos_idx = proba.shape[1] - 1
    pos_proba = proba[:, pos_idx]
    y_pred    = (pos_proba >= 0.5).astype(int)
    return y_pred, pos_proba


def risk_label(p: float):
    if p >= 0.70:
        return "High Risk",     "danger",  "🔴"
    elif p >= 0.40:
        return "Moderate Risk", "warning", "🟡"
    else:
        return "Low Risk",      "success", "🟢"


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CardioRisk AI")
    st.markdown("**XGBoost + Platt Scaling**")
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("""
- **Algorithm:** XGBoost (tuned)
- **Calibration:** Platt Scaling (cv=10)
- **Validation:** 10-Fold Stratified CV
- **ROC AUC:** 0.9977 ± 0.0039
- **Sensitivity:** 98.77%
- **Specificity:** 97.06%
- **Brier Score:** 0.0137
    """)
    st.markdown("---")
    st.markdown("### Features Used")
    for f in selected_features:
        st.markdown(f"- `{f}`")
    st.markdown("---")
    st.caption("⚠️ For research use only. Not a substitute for clinical diagnosis.")


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:28px 0 12px 0'>
    <h1 style='margin:0;font-size:32px;font-weight:800;color:#0f172a'>
        🫀 CardioRisk AI
    </h1>
    <p style='margin:4px 0 0 0;color:#64748b;font-size:16px'>
        Heart Disease Risk Prediction — Powered by Tuned XGBoost with Platt Calibration
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["🧍 Single Patient Prediction", "🗂️ Batch Prediction (CSV)"])


# ════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════
with tab1:

    st.markdown("<div class='section-title'>Patient Demographics</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=1.0,
                               max_value=120.0, value=45.0, step=1.0)
    with c2:
        gender_label = st.radio("Gender", ["Female", "Male"],
                                 index=1, horizontal=True)
        gender = 1.0 if gender_label == "Male" else 0.0
    with c3:
        heart_rate = st.number_input("Heart Rate (bpm)",
                                      min_value=30.0, max_value=220.0,
                                      value=72.0, step=1.0)

    st.markdown("<div class='section-title' style='margin-top:16px'>Blood Pressure</div>",
                unsafe_allow_html=True)
    c4, c5, _ = st.columns(3)
    with c4:
        sbp = st.number_input("Systolic BP (mmHg)",
                               min_value=70.0, max_value=250.0,
                               value=120.0, step=1.0)
    with c5:
        dbp = st.number_input("Diastolic BP (mmHg)",
                               min_value=40.0, max_value=150.0,
                               value=80.0, step=1.0)
    if sbp < dbp:
        st.warning("⚠️ Systolic BP should be greater than Diastolic BP.")

    st.markdown("<div class='section-title' style='margin-top:16px'>Laboratory Values</div>",
                unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    with c6:
        blood_sugar = st.number_input("Blood Sugar (mmol/L)",
                                       min_value=0.0, max_value=30.0,
                                       value=5.5, step=0.1)
    with c7:
        ckmb = st.number_input("CK-MB (U/L)",
                                min_value=0.0, max_value=100.0,
                                value=1.0, step=0.1)
    with c8:
        troponin = st.number_input(
            "Troponin (ng/mL)",
            min_value=0.000,
            max_value=50.0,
            value=0.001,
            step=0.001,
            format="%.3f"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button("🔍 Analyse Risk")

    if predict_btn:
        single_input = {
            "Age"                      : age,
            "Gender"                   : gender,
            "Heart rate"               : heart_rate,
            "Systolic blood pressure"  : sbp,
            "Diastolic blood pressure" : dbp,
            "Blood sugar"              : blood_sugar,
            "CK-MB"                    : ckmb,
            "Troponin"                 : troponin,
        }
        df_single = pd.DataFrame([single_input])

        try:
            y_pred, proba = predict_df(df_single)
            p             = float(proba[0])
            label, card_cls, icon = risk_label(p)
            pct           = p * 100

            st.markdown("---")
            st.markdown("<div class='section-title'>Prediction Result</div>",
                        unsafe_allow_html=True)

            r1c, r2c, r3c = st.columns([1.2, 1, 1])

            # Left — diagnosis banner
            with r1c:
                if y_pred[0] == 1:
                    st.markdown("""
                    <div class='result-positive'>
                        <div style='font-size:48px'>🚨</div>
                        <div style='font-size:22px;font-weight:800;
                                    color:#dc2626;margin-top:8px'>
                            Heart Disease Detected
                        </div>
                        <div style='color:#991b1b;margin-top:4px;font-size:14px'>
                            Please seek immediate clinical evaluation
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='result-negative'>
                        <div style='font-size:48px'>✅</div>
                        <div style='font-size:22px;font-weight:800;
                                    color:#16a34a;margin-top:8px'>
                            No Heart Disease Detected
                        </div>
                        <div style='color:#166534;margin-top:4px;font-size:14px'>
                            Continue routine monitoring
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Centre — risk level + probability bar using Streamlit native
            with r2c:
                st.markdown(f"""
                <div class='metric-card {card_cls}'>
                    <div style='font-size:12px;color:#6b7280;
                                font-weight:600;text-transform:uppercase'>
                        Risk Level
                    </div>
                    <div style='font-size:28px;font-weight:800;
                                margin-top:4px;color:#0f172a'>
                        {icon} {label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:12px;color:#6b7280;
                                font-weight:600;text-transform:uppercase'>
                        Disease Probability
                    </div>
                    <div style='font-size:32px;font-weight:800;
                                color:#0f172a;margin-top:4px'>
                        {pct:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(p)

            # Right — confidence + prediction
            with r3c:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:12px;color:#6b7280;
                                font-weight:600;text-transform:uppercase'>
                        Model Confidence
                    </div>
                    <div style='font-size:28px;font-weight:800;
                                color:#0f172a;margin-top:4px'>
                        {max(pct, 100 - pct):.1f}%
                    </div>
                    <div style='color:#6b7280;font-size:13px;margin-top:4px'>
                        Calibrated via Platt Scaling
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:12px;color:#6b7280;
                                font-weight:600;text-transform:uppercase'>
                        Binary Prediction
                    </div>
                    <div style='font-size:20px;font-weight:700;
                                color:#0f172a;margin-top:4px'>
                        {"Positive 🔴" if y_pred[0] == 1 else "Negative 🟢"}
                    </div>
                    <div style='color:#6b7280;font-size:13px;margin-top:4px'>
                        Decision threshold: 0.50
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


# ════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='metric-card'>
        <div style='font-weight:700;font-size:15px;color:#0f172a'>
            📋 CSV Format Requirements
        </div>
        <div style='color:#64748b;font-size:13px;margin-top:6px'>
            Your CSV must include the following columns:
            <code>Age, Gender, Heart rate, Systolic blood pressure,
            Diastolic blood pressure, Blood sugar, CK-MB, Troponin</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    file = st.file_uploader("Upload your CSV file", type=["csv"],
                             help="Max file size: 200MB")
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
            st.markdown(f"**{len(df_raw):,} records loaded** — preview below:")
            st.dataframe(df_raw.head(5), use_container_width=True)

            with st.spinner("Running predictions..."):
                y_pred, proba = predict_df(df_raw)

            out = df_raw.copy()
            out["Prediction"]           = ["Heart Disease" if p == 1
                                            else "No Heart Disease" for p in y_pred]
            out["Risk Probability (%)"] = (proba * 100).round(2)
            out["Risk Level"]           = [risk_label(p)[0] for p in proba]

            st.markdown("---")
            st.markdown("### Results")

            total    = len(out)
            positive = int(y_pred.sum())
            negative = total - positive

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Patients",         f"{total:,}")
            m2.metric("Heart Disease Detected", f"{positive:,}",
                      delta=f"{positive / total * 100:.1f}%")
            m3.metric("No Disease",             f"{negative:,}",
                      delta=f"{negative / total * 100:.1f}%")

            st.dataframe(out, use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_bytes,
                file_name="cardiorisk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")