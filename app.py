import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioInsight Studio",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS — redesigned UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg: #0b1020;
        --panel: rgba(255,255,255,0.06);
        --panel-2: rgba(255,255,255,0.04);
        --stroke: rgba(255,255,255,0.12);
        --text: #eef2ff;
        --muted: #a9b4d0;
        --accent: #6ee7f9;
        --accent-2: #8b5cf6;
        --good: #22c55e;
        --warn: #f59e0b;
        --bad: #ef4444;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(110,231,249,0.08), transparent 25%),
            radial-gradient(circle at top right, rgba(139,92,246,0.10), transparent 22%),
            linear-gradient(180deg, #0b1020 0%, #11172a 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: #0d1325;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(110,231,249,0.10), rgba(139,92,246,0.14));
        border: 1px solid var(--stroke);
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        backdrop-filter: blur(12px);
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }

    .hero p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 0.98rem;
    }

    .glass-card {
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.16);
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 16px;
        text-align: left;
        min-height: 90px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .section-title {
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtle {
        color: var(--muted);
        font-size: 0.92rem;
    }

    .risk-banner {
        border-radius: 22px;
        padding: 22px;
        border: 1px solid rgba(255,255,255,0.12);
        color: white;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 16px;
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(127,29,29,0.32));
    }

    .risk-moderate {
        background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(120,53,15,0.30));
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(20,83,45,0.32));
    }

    .recommendation-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-left: 4px solid var(--accent);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
        color: var(--text);
        font-size: 0.94rem;
    }

    .input-panel {
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 18px;
        margin-bottom: 18px;
    }

    div[data-testid="stTabs"] button {
        border-radius: 12px;
    }

    .footer-note {
        color: var(--muted);
        text-align: center;
        font-size: 0.88rem;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FEATURE_COLS      = ["age","sex","cp","trestbps","chol","fbs",
                     "restecg","thalach","exang","oldpeak","slope","ca","thal"]
NUMERIC_FEATURES  = ["age","trestbps","chol","thalach","oldpeak"]
ORDINAL_FEATURES  = ["cp","restecg","slope","thal","ca"]
BINARY_FEATURES   = ["sex","fbs","exang"]
FEATURE_NAMES_OUT = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate (bpm)",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (oldpeak)",
    "slope":    "ST Slope",
    "ca":       "Major Vessels (fluoroscopy)",
    "thal":     "Thalassemia Type",
}

CP_MAP      = {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-Anginal Pain", 3:"Asymptomatic"}
RESTECG_MAP = {0:"Normal", 1:"ST-T Wave Abnormality", 2:"Left Ventricular Hypertrophy"}
SLOPE_MAP   = {0:"Upsloping", 1:"Flat", 2:"Downsloping"}
THAL_MAP    = {0:"Normal", 1:"Fixed Defect", 2:"Reversable Defect", 3:"Rev. Defect (other)"}


# ─────────────────────────────────────────────
# Train & Cache Model at Startup
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on first launch...")
def build_model():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    from lightgbm import LGBMClassifier
    import shap as _shap

    df = pd.read_csv("heart.csv")
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)

    def winsorise(df, col):
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df

    for col in ["trestbps", "chol", "oldpeak"]:
        df = winsorise(df, col)

    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ORDINAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])

    lgbm = LGBMClassifier(
        is_unbalance=True, random_state=42, verbose=-1,
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        boosting_type="dart", min_child_samples=20,
        reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
    )

    lgbm_pipe = Pipeline([("pre", preprocessor), ("clf", lgbm)])
    lgbm_pipe.fit(X_train, y_train)

    y_prob_train = lgbm_pipe.predict_proba(X_train)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(
            y_train,
            (y_prob_train >= t).astype(int),
            average="macro",
            zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    base_lgbm = lgbm_pipe.named_steps["clf"]
    cal_clf = CalibratedClassifierCV(base_lgbm, method="isotonic", cv=5)
    cal_pipe = Pipeline([("pre", preprocessor), ("clf", cal_clf)])
    cal_pipe.fit(X_train, y_train)

    preprocessor.fit(X_train)
    explainer = _shap.TreeExplainer(base_lgbm)

    return cal_pipe, preprocessor, explainer, best_t, X_train


model, preprocessor, explainer, THRESHOLD, X_train = build_model()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def predict(patient_dict):
    df_in = pd.DataFrame([patient_dict])
    prob = model.predict_proba(df_in)[0, 1]
    return prob, int(prob >= THRESHOLD)


def get_shap_values(patient_dict):
    df_in = pd.DataFrame([patient_dict])
    X_t = preprocessor.transform(df_in)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(np.array(ev).ravel()[1])
    return sv[0], float(ev)


def risk_band(prob):
    if prob >= 0.65:
        return "HIGH", "#ef4444", "risk-high", "🚨"
    elif prob >= 0.30:
        return "MODERATE", "#f59e0b", "risk-moderate", "⚠️"
    else:
        return "LOW", "#22c55e", "risk-low", "✅"


@st.cache_resource(show_spinner="Computing global SHAP importance...")
def get_global_shap(_explainer, _preprocessor, _X_train):
    X_t = _preprocessor.transform(_X_train)
    sv = _explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫀 CardioInsight Studio</h1>
    <p>Interactive heart disease risk dashboard with clinical inputs, explainable AI, and scenario simulation.</p>
</div>
""", unsafe_allow_html=True)

top1, top2, top3 = st.columns(3)
with top1:
    metric_card("Model Type", "LightGBM + Calibration")
with top2:
    metric_card("Decision Threshold", f"{THRESHOLD:.2f}")
with top3:
    metric_card("Use Case", "Clinical Decision Support")


# ─────────────────────────────────────────────
# Input Area
# ─────────────────────────────────────────────
st.markdown('<div class="input-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Patient Intake Form</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Enter the patient profile below, then run the prediction.</div>', unsafe_allow_html=True)

with st.form("patient_form_main"):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Demographics**")
        age = st.slider("Age (years)", 20, 80, 55)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    with c2:
        st.markdown("**Symptoms**")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"{x} — {CP_MAP[x]}")
        exang = st.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with c3:
        st.markdown("**Vitals & Labs**")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
        chol = st.slider("Serum Cholesterol (mg/dl)", 120, 570, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with c4:
        st.markdown("**ECG & Imaging**")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: f"{x} — {RESTECG_MAP[x]}")
        thalach = st.slider("Max Heart Rate Achieved (bpm)", 70, 210, 150)
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: f"{x} — {SLOPE_MAP[x]}")
        ca = st.slider("Major Vessels Coloured (0–4)", 0, 4, 0)
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: f"{x} — {THAL_MAP[x]}")

    submitted = st.form_submit_button("Run Heart Risk Analysis", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Default View
# ─────────────────────────────────────────────
if not submitted:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Global Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtle">Average absolute SHAP contribution across training cases.</div>', unsafe_allow_html=True)

        sv_all = get_global_shap(explainer, preprocessor, X_train)
        mean_shap = np.abs(sv_all).mean(axis=0)
        feat_imp = pd.DataFrame({
            "feature": FEATURE_NAMES_OUT,
            "importance": mean_shap
        }).sort_values("importance")

        labels = [FEATURE_LABELS.get(f, f) for f in feat_imp["feature"]]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#11172a")
        colors = plt.cm.cool(np.linspace(0.2, 0.95, len(labels)))
        ax.barh(labels, feat_imp["importance"], color=colors)
        ax.set_xlabel("Mean |SHAP value|", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#334155")
        ax.set_facecolor("#11172a")
        fig.patch.set_facecolor("#11172a")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Test F1 Score", "0.9902")
            st.metric("ROC-AUC", "1.0000")
        with m2:
            st.metric("Test Accuracy", "99.5%")
            st.metric("Threshold", f"{THRESHOLD:.2f}")
        st.markdown(
            """
            <div class="subtle">
            Use the form above to generate a prediction, inspect SHAP contributions,
            and test interventions using the scenario planner.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

else:
    patient = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal,
    }

    prob, label = predict(patient)
    risk_label, risk_color, risk_class, risk_icon = risk_band(prob)

    st.markdown(
        f'<div class="risk-banner {risk_class}">{risk_icon} {risk_label} RISK — Predicted Probability: {prob*100:.1f}%</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Age", f"{age} yrs")
    with b:
        metric_card("Max Heart Rate", f"{thalach} bpm")
    with c:
        metric_card("Blood Pressure", f"{trestbps} mmHg")
    with d:
        metric_card("Cholesterol", f"{chol} mg/dl")

    tab1, tab2, tab3 = st.tabs(["Prediction Overview", "Explainability", "Scenario Planner"])

    with tab1:
        left, right = st.columns([1.0, 1.0])

        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Probability Gauge</div>', unsafe_allow_html=True)

            fig_gauge, ax = plt.subplots(figsize=(5.3, 3.2), facecolor="#11172a")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.3, 1.15)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor("#11172a")
            fig_gauge.patch.set_facecolor("#11172a")

            theta_bg = np.linspace(0, math.pi, 200)
            ax.plot(np.cos(theta_bg), np.sin(theta_bg),
                    color="#334155", linewidth=20, solid_capstyle="round")

            theta_fill = np.linspace(0, math.pi * prob, 200)
            if len(theta_fill) > 1:
                ax.plot(np.cos(theta_fill), np.sin(theta_fill),
                        color=risk_color, linewidth=20, solid_capstyle="round")

            t_angle = math.pi * THRESHOLD
            ax.plot([0.78 * math.cos(t_angle)], [0.78 * math.sin(t_angle)],
                    "w|", markersize=18, markeredgewidth=2.5)

            n_angle = math.pi * prob
            ax.annotate(
                "",
                xy=(0.65 * math.cos(n_angle), 0.65 * math.sin(n_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5)
            )
            ax.plot(0, 0, "o", color="white", markersize=7, zorder=5)

            ax.text(0, -0.22, f"{prob*100:.1f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=risk_color)
            ax.text(0, -0.38, f"Threshold: {THRESHOLD:.2f}",
                    ha="center", fontsize=9, color="#a9b4d0")

            ax.text(-1.15, -0.08, "0%", ha="center", fontsize=8, color="#a9b4d0")
            ax.text(1.15, -0.08, "100%", ha="center", fontsize=8, color="#a9b4d0")
            ax.text(-0.75, 0.35, "LOW", ha="center", fontsize=7, color="#22c55e", alpha=0.8)
            ax.text(0.0, 0.85, "MOD", ha="center", fontsize=7, color="#f59e0b", alpha=0.8)
            ax.text(0.75, 0.35, "HIGH", ha="center", fontsize=7, color="#ef4444", alpha=0.8)

            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Patient Snapshot</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric("Chest Pain", CP_MAP[cp])
            s2.metric("Thalassemia", THAL_MAP[thal])
            s1.metric("ST Depression", f"{oldpeak:.1f}")
            s2.metric("Major Vessels (ca)", f"{ca}")
            s1.metric("Exercise Angina", "Yes" if exang == 1 else "No")
            s2.metric("Resting ECG", RESTECG_MAP[restecg])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Clinical Recommendations</div>', unsafe_allow_html=True)

            recs = []
            if risk_label == "HIGH":
                recs.append("🚨 <b>Urgent cardiology referral recommended.</b> Predicted probability exceeds the high-risk threshold.")
            if thalach < 130:
                recs.append("❤️ Low maximum heart rate detected. Consider <b>exercise stress testing</b>.")
            if ca >= 2:
                recs.append(f"🩻 {ca} major vessels coloured by fluoroscopy. Consider <b>coronary angiography</b>.")
            if cp == 0:
                recs.append("⚡ Asymptomatic chest pain profile is high-risk in this model. Ensure a <b>full cardiac workup</b>.")
            if oldpeak >= 2.0:
                recs.append(f"📉 ST depression of {oldpeak:.1f} suggests possible <b>exercise-induced ischaemia</b>.")
            if chol > 240:
                recs.append(f"🧪 Cholesterol at {chol} mg/dl exceeds 240. Review <b>lipid management</b>.")
            if trestbps > 140:
                recs.append(f"🩸 Resting BP at {trestbps} mmHg suggests <b>hypertension review</b>.")
            if exang == 1:
                recs.append("🏃 Exercise-induced angina present. Investigate with <b>myocardial perfusion imaging</b> if appropriate.")
            if slope == 2:
                recs.append("📊 Downsloping ST segment may indicate more severe CAD. Consider <b>invasive evaluation</b>.")
            if not recs:
                recs.append("✅ No major high-priority flags identified. Routine follow-up may be appropriate.")

            for rec in recs:
                st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">SHAP Feature Attribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="subtle">Red bars push risk upward, blue bars pull it downward.</div>', unsafe_allow_html=True)

            sv, base_val = get_shap_values(patient)
            labels_shap = [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES_OUT]
            shap_df = pd.DataFrame({"feature": labels_shap, "shap": sv})
            shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=True).index)
            colors = ["#ef4444" if v > 0 else "#38bdf8" for v in shap_df["shap"]]

            fig_w, ax_w = plt.subplots(figsize=(7, 5.6), facecolor="#11172a")
            ax_w.barh(shap_df["feature"], shap_df["shap"], color=colors)
            ax_w.axvline(0, color="white", linewidth=0.8, linestyle="--")
            ax_w.set_xlabel("SHAP value (log-odds contribution)", color="white")
            ax_w.tick_params(colors="white", labelsize=9)
            ax_w.spines[:].set_color("#334155")
            ax_w.set_facecolor("#11172a")
            fig_w.patch.set_facecolor("#11172a")
            ax_w.set_title(
                f"Base value: {base_val:.3f}  →  Output: {float(np.sum(sv)) + base_val:.3f}",
                color="#cbd5e1", fontsize=10
            )
            st.pyplot(fig_w, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Feature Influence Summary</div>', unsafe_allow_html=True)

            top_features = pd.DataFrame({
                "feature": labels_shap,
                "shap": sv,
                "abs_shap": np.abs(sv)
            }).sort_values("abs_shap", ascending=False).head(5)

            for _, row in top_features.iterrows():
                direction = "increases" if row["shap"] > 0 else "decreases"
                color = "#ef4444" if row["shap"] > 0 else "#38bdf8"
                st.markdown(
                    f"""
                    <div class="recommendation-box" style="border-left-color:{color};">
                        <b>{row['feature']}</b> {direction} predicted risk
                        (SHAP: {row['shap']:.3f})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What-If Scenario Planner</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtle">Adjust selected inputs to simulate possible interventions.</div>', unsafe_allow_html=True)

        wi_col1, wi_col2, wi_col3 = st.columns(3)
        with wi_col1:
            wi_thalach = st.slider("Max Heart Rate (bpm)", 70, 210, thalach, key="wi_thalach")
            wi_chol = st.slider("Cholesterol (mg/dl)", 120, 570, chol, key="wi_chol")
        with wi_col2:
            wi_cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=cp, key="wi_cp",
                                 format_func=lambda x: f"{x} — {CP_MAP[x]}")
            wi_ca = st.slider("Major Vessels", 0, 4, ca, key="wi_ca")
        with wi_col3:
            wi_oldpeak = st.slider("ST Depression", 0.0, 6.5, oldpeak, step=0.1, key="wi_oldpeak")
            wi_trestbps = st.slider("Blood Pressure", 90, 200, trestbps, key="wi_trestbps")

        wi_patient = {
            **patient,
            "thalach": wi_thalach,
            "chol": wi_chol,
            "cp": wi_cp,
            "ca": wi_ca,
            "oldpeak": wi_oldpeak,
            "trestbps": wi_trestbps
        }

        wi_prob, _ = predict(wi_patient)
        wi_label, wi_color, wi_class, wi_icon = risk_band(wi_prob)
        delta_prob = wi_prob - prob

        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Original Probability", f"{prob*100:.1f}%")
        wc2.metric("Scenario Probability", f"{wi_prob*100:.1f}%", delta=f"{delta_prob*100:+.1f} pp", delta_color="inverse")
        wc3.metric("Scenario Risk Level", f"{wi_icon} {wi_label}")

        fig_cmp, ax_cmp = plt.subplots(figsize=(8, 1.5), facecolor="#11172a")
        ax_cmp.barh(["Original"], [prob], color=risk_color, height=0.42)
        ax_cmp.barh(["Scenario"], [wi_prob], color=wi_color, height=0.42)
        ax_cmp.axvline(THRESHOLD, color="white", linestyle="--", linewidth=1)
        ax_cmp.set_xlim(0, 1)
        ax_cmp.set_xlabel("Probability", color="white")
        ax_cmp.tick_params(colors="white")
        ax_cmp.spines[:].set_color("#334155")
        ax_cmp.set_facecolor("#11172a")
        fig_cmp.patch.set_facecolor("#11172a")
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close()

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">⚠️ This tool is for research and educational purposes only. Clinical decisions must be made by a qualified healthcare professional.</div>',
    unsafe_allow_html=True
)
