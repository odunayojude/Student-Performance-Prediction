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
    # CREATE EMPTY DATAFRAME
    # =========================
    input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    # =========================
    # SET NUMERIC FEATURES
    # =========================
    def set_if_exists(col, value):
        if col in input_df.columns:
            input_df[col] = value

    set_if_exists("raisedhands", raisedhands)
    set_if_exists("VisITedResources", visited)
    set_if_exists("Discussion", discussion)
    set_if_exists("AnnouncementsView", announcements)

    # =========================
    # SCALE INPUT (IMPORTANT!)
    # =========================
    input_scaled = scaler.transform(input_df)

    # =========================
    # PREDICT
    # =========================
    prediction = model.predict(input_scaled)[0]

    # =========================
    # MAP OUTPUT LABELS (based on LabelEncoder)
    # =========================
    label_map = {0: "Low", 1: "Medium", 2: "High"}

    st.success(f"Prediction: {label_map.get(prediction, prediction)}")