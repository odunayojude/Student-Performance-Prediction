import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Predictor", layout="wide")

# =========================
# LOAD MODELS
# =========================
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# =========================
# FIXED SPLIT SCREEN CSS
# =========================
st.markdown(
    """
    <style>

    /* Remove default padding */
    .block-container {
        padding: 0rem 1rem;
    }

    /* Make columns full height */
    section.main > div {
        padding-top: 0rem;
    }

    /* LEFT PANEL */
    div[data-testid="column"]:nth-child(1) {
        position: fixed;
        left: 0;
        top: 0;
        width: 45%;
        height: 100vh;
        background-color: #0e1117;
        padding: 2rem;
        overflow: hidden;
        border-right: 1px solid #333;
    }

    /* RIGHT PANEL */
    div[data-testid="column"]:nth-child(2) {
        margin-left: 45%;
        width: 55%;
        padding: 2rem;
        height: 100vh;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1.2])

# =========================
# LEFT PANEL - PREDICTOR (FIXED)
# =========================
with col1:
    st.markdown("## 📥 Predictor Panel")

    raisedhands = st.slider("Raised Hands", 0, 100, 50)
    visited = st.slider("Visited Resources", 0, 100, 50)
    discussion = st.slider("Discussion Activity", 0, 100, 50)
    announcements = st.slider("Announcements View", 0, 100, 50)

    predict_btn = st.button("🚀 Predict")

# =========================
# RIGHT PANEL - OUTPUT (FIXED)
# =========================
with col2:
    st.markdown("## 📊 Output Panel")

    if predict_btn:

        # =========================
        # BUILD INPUT
        # =========================
        input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

        def set_if_exists(col, value):
            if col in input_df.columns:
                input_df[col] = value

        set_if_exists("raisedhands", raisedhands)
        set_if_exists("VisITedResources", visited)
        set_if_exists("Discussion", discussion)
        set_if_exists("AnnouncementsView", announcements)

        # =========================
        # SCALE + PREDICT
        # =========================
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        labels = ["Low", "Medium", "High"]
        predicted_label = labels[prediction]
        confidence = np.max(probabilities)

        # =========================
        # RESULT
        # =========================
        if predicted_label == "High":
            st.success("🟢 High Performance")
        elif predicted_label == "Medium":
            st.warning("🟡 Medium Performance")
        else:
            st.error("🔴 Low Performance")

        st.metric("Confidence", f"{confidence:.2f}")

        # =========================
        # CHART
        # =========================
        st.markdown("### 📈 Probabilities")

        prob_df = pd.DataFrame({
            "Class": labels,
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Class"))

        # =========================
        # INSIGHT
        # =========================
        st.markdown("### 🧠 Insight")

        if predicted_label == "High":
            st.info("Strong student performance detected.")
        elif predicted_label == "Medium":
            st.info("Moderate performance. Encourage more engagement.")
        else:
            st.info("Low performance detected. Intervention needed.")

    else:
        st.info("Enter values on the left and click Predict.")