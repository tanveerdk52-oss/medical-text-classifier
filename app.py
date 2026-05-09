
import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Medical Text Classifier",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Medical Text Classification")
st.markdown("**Classify medical questions into specialties using AI**")
st.markdown("---")

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("lr_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    classes = np.load("label_classes.npy", allow_pickle=True)
    return model, tfidf, classes

model, tfidf, classes = load_model()

# Example inputs
st.subheader("📋 Try an Example")
examples = {
    "Cardiology": "Patient presents with chest pain and shortness of breath. ECG shows ST elevation. Troponin 2.4 ng/mL elevated.",
    "Neurology":  "45 year old with sudden onset weakness on right side. CT brain shows hypodense area in left MCA territory.",
    "Surgery":    "Patient has acute abdominal pain with rebound tenderness. WBC 15000. CT confirms appendicitis.",
    "Pharmacology": "Which drug is used for treatment of hypertension and also has cardioprotective effects in heart failure patients?",
    "Pediatrics": "6 year old child with fever 39.5C for 3 days. Throat culture positive for streptococcus."
}

col1, col2, col3 = st.columns(3)
if col1.button("🫀 Cardiology"):
    st.session_state.input_text = examples["Cardiology"]
if col2.button("🧠 Neurology"):
    st.session_state.input_text = examples["Neurology"]
if col3.button("🔪 Surgery"):
    st.session_state.input_text = examples["Surgery"]

col4, col5 = st.columns(2)
if col4.button("💊 Pharmacology"):
    st.session_state.input_text = examples["Pharmacology"]
if col5.button("👶 Pediatrics"):
    st.session_state.input_text = examples["Pediatrics"]

st.markdown("---")

# Text input
st.subheader("✍️ Enter Medical Text")
input_text = st.text_area(
    "Type or paste medical text here:",
    value=st.session_state.get("input_text", ""),
    height=150,
    placeholder="Enter a medical question or clinical note..."
)

# Predict button
if st.button("🔍 Classify", use_container_width=True):
    if input_text.strip():
        # Preprocess
        import re
        text = input_text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)

        # Predict
        vec   = tfidf.transform([text])
        pred  = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        predicted_label = classes[pred]
        confidence      = proba[pred] * 100

        # Show result
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        st.success(f"**Predicted Specialty: {predicted_label}**")
        st.metric("Confidence", f"{confidence:.1f}%")

        # Top 5 predictions
        st.subheader("📊 Top 5 Predictions")
        top5_idx = np.argsort(proba)[::-1][:5]
        for i, idx in enumerate(top5_idx):
            bar_val = proba[idx]
            st.write(f"**{i+1}. {classes[idx]}** — {proba[idx]*100:.1f}%")
            st.progress(float(bar_val))
    else:
        st.warning("Please enter some medical text first!")

# Footer
st.markdown("---")
st.markdown("**DS Project — Group 29 | Medical Text Classification**")
st.markdown("Models: Logistic Regression | BiLSTM | DistilBERT")
