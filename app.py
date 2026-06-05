import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Student Performance Prediction")

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# =========================
# USER INPUTS (ONLY REAL FEATURES)
# =========================
raisedhands = st.slider("Raised Hands", 0, 100, 50)
visited = st.slider("Visited Resources", 0, 100, 50)
discussion = st.slider("Discussion Activity", 0, 100, 50)
announcements = st.slider("Announcements View", 0, 100, 50)

if st.button("Predict"):

    # =========================
    # CREATE INPUT FRAME
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
    # SCALE INPUT
    # =========================
    input_scaled = scaler.transform(input_df)

    # =========================
    # PREDICT CLASS + PROBABILITY
    # =========================
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Class labels (based on your training encoder)
    class_labels = ["Low", "Medium", "High"]

    predicted_label = class_labels[prediction]
    confidence = np.max(probabilities)

    # =========================
    # DISPLAY MAIN RESULT
    # =========================
    st.markdown("## 🎯 Prediction Result")

    if predicted_label == "High":
        st.success(f"🟢 Performance: {predicted_label}")
    elif predicted_label == "Medium":
        st.warning(f"🟡 Performance: {predicted_label}")
    else:
        st.error(f"🔴 Performance: {predicted_label}")

    st.write(f"**Confidence:** {confidence:.2f}")

    # =========================
    # PROBABILITY BREAKDOWN
    # =========================
    st.markdown("## 📊 Class Probabilities")

    prob_df = pd.DataFrame({
        "Performance Level": class_labels,
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Performance Level"))

    # =========================
    # INTERPRETATION MESSAGE
    # =========================
    st.markdown("## 🧠 Insight")

    if predicted_label == "High":
        st.info("This student is performing very well. Keep reinforcing current study habits.")
    elif predicted_label == "Medium":
        st.info("This student is doing okay but has room for improvement in engagement.")
    else:
        st.info("This student may need academic support and increased engagement strategies.")