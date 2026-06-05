import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")  # ⭐ IMPORTANT: full screen layout

st.title("🎓 Student Performance Prediction Dashboard")

# =========================
# LOAD MODELS
# =========================
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# =========================
# CREATE 2 COLUMNS
# =========================
col1, col2 = st.columns([1, 1.2])  # right side slightly bigger

# =========================
# LEFT SIDE: INPUTS
# =========================
with col1:
    st.markdown("## 📥 Student Inputs")

    raisedhands = st.slider("Raised Hands", 0, 100, 50)
    visited = st.slider("Visited Resources", 0, 100, 50)
    discussion = st.slider("Discussion Activity", 0, 100, 50)
    announcements = st.slider("Announcements View", 0, 100, 50)

    predict_btn = st.button("🚀 Predict Performance")

# =========================
# RIGHT SIDE: OUTPUT
# =========================
with col2:

    st.markdown("## 📊 Prediction Result")

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
            st.success(f"🟢 {predicted_label} Performance")
        elif predicted_label == "Medium":
            st.warning(f"🟡 {predicted_label} Performance")
        else:
            st.error(f"🔴 {predicted_label} Performance")

        st.metric("Confidence Score", f"{confidence:.2f}")

        # =========================
        # CHART
        # =========================
        st.markdown("### 📈 Probability Breakdown")

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
            st.info("Strong academic engagement detected.")
        elif predicted_label == "Medium":
            st.info("Moderate performance. Some improvement needed.")
        else:
            st.info("Low engagement detected. Student may need support.")