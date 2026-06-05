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
# CSS (SCROLL + RESPONSIVE)
# =========================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
        }

        .left-panel, .right-panel {
            height: 85vh;
            overflow-y: auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #0e1117;
        }

        .right-panel {
            border-left: 1px solid #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LAYOUT (2 COLUMNS)
# =========================
col1, col2 = st.columns([1, 1.2], gap="large")

# =========================
# LEFT PANEL - INPUTS
# =========================
with col1:
    st.markdown("## 📥 Predictor", unsafe_allow_html=True)

    with st.container():
        raisedhands = st.slider("Raised Hands", 0, 100, 50)
        visited = st.slider("Visited Resources", 0, 100, 50)
        discussion = st.slider("Discussion Activity", 0, 100, 50)
        announcements = st.slider("Announcements View", 0, 100, 50)

        predict_btn = st.button("🚀 Predict Performance")

# =========================
# RIGHT PANEL - OUTPUT
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
        # RESULT DISPLAY
        # =========================
        if predicted_label == "High":
            st.success(f"🟢 High Performance")
        elif predicted_label == "Medium":
            st.warning(f"🟡 Medium Performance")
        else:
            st.error(f"🔴 Low Performance")

        st.metric("Confidence Score", f"{confidence:.2f}")

        # =========================
        # PROBABILITY CHART
        # =========================
        st.markdown("### 📈 Probability Distribution")

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
            st.info("Student shows strong engagement and consistent learning behavior.")
        elif predicted_label == "Medium":
            st.info("Moderate performance. Encourage more participation.")
        else:
            st.info("Low engagement detected. Intervention recommended.")
    else:
        st.info("👈 Enter values on the left and click Predict to see results.")