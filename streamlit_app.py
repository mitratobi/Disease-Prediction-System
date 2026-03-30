import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.predict import predict, get_symptom_list

st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Disease Prediction System")
st.markdown("Select up to **5 symptoms** and get an instant disease prediction.")

symptoms = get_symptom_list()
display = [s.replace("_", " ").title() for s in symptoms]
label_to_raw = dict(zip(display, symptoms))

st.markdown("---")

selected = st.multiselect(
    "Choose your symptoms (max 5):",
    options=display,
    max_selections=5,
    placeholder="Start typing a symptom...",
)

if st.button("Predict Disease", type="primary", disabled=len(selected) == 0):
    raw_selected = [label_to_raw[s] for s in selected]
    with st.spinner("Analysing symptoms..."):
        result = predict(raw_selected)

    st.success(f"**Predicted Disease:** {result}")
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational purposes only. "
        "Always consult a qualified medical professional for diagnosis."
    )
