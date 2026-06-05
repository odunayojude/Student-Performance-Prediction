import streamlit as st
import joblib

st.title("Student Performance Prediction")

model = joblib.load("models/random_forest.pkl")

raisedhands = st.slider("Raised Hands", 0, 100, 50)
visited = st.slider("Visited Resources", 0, 100, 50)
discussion = st.slider("Discussion Activity", 0, 100, 50)

if st.button("Predict"):
    st.success("Prediction functionality ready")