"""
app.py
======
Section 5 - Final Application Plan

Streamlit web application: user enters Instagram profile details, the app
runs the same NLP preprocessing + feature extraction used at training time,
loads the best trained model, and displays a Real/Fake prediction with a
confidence score. Predictions below a 70% confidence threshold are shown
as "Uncertain" (Section 7, Q15).

Run with:
    streamlit run src/app.py
"""

import os
import sys
import joblib
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import verbalize_row
from src.text_cleaning import preprocess_for_bow_lstm, preprocess_for_tfidf_rnn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CONFIDENCE_THRESHOLD = 0.70

st.set_page_config(page_title="Fake Instagram Account Detector", page_icon="🕵️", layout="centered")


@st.cache_resource
def load_himas_model():
    """Best ML fallback: Logistic Regression on BoW + numeric features."""
    try:
        lr = joblib.load(os.path.join(MODELS_DIR, "himas_logistic_regression.pkl"))
        vec = joblib.load(os.path.join(MODELS_DIR, "himas_bow_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "himas_scaler.pkl"))
        return lr, vec, scaler
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def load_afrith_model():
    """Alternative ML model: Random Forest on TF-IDF + numeric features."""
    try:
        rf = joblib.load(os.path.join(MODELS_DIR, "afrith_random_forest.pkl"))
        vec = joblib.load(os.path.join(MODELS_DIR, "afrith_tfidf_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "afrith_scaler.pkl"))
        return rf, vec, scaler
    except FileNotFoundError:
        return None, None, None


def predict_logistic_regression(row_dict, lr, vec, scaler):
    text = verbalize_row(row_dict)
    cleaned = preprocess_for_bow_lstm(text)
    X_text = vec.transform([cleaned])
    X_num = scaler.transform([[
        row_dict["profile pic"], row_dict["nums/length username"],
        row_dict["fullname words"], row_dict["nums/length fullname"],
        row_dict["name==username"], row_dict["description length"],
        row_dict["external URL"], row_dict["private"],
        row_dict["#posts"], row_dict["#followers"], row_dict["#follows"],
        row_dict["followers_following_ratio"],
    ]])
    X = hstack([X_text, csr_matrix(X_num)])
    proba = lr.predict_proba(X)[0, 1]
    return proba, text


def predict_random_forest(row_dict, rf, vec, scaler):
    text = verbalize_row(row_dict)
    cleaned = preprocess_for_tfidf_rnn(text)
    X_text = vec.transform([cleaned])
    X_num = scaler.transform([[
        row_dict["profile pic"], row_dict["nums/length username"],
        row_dict["fullname words"], row_dict["nums/length fullname"],
        row_dict["name==username"], row_dict["description length"],
        row_dict["external URL"], row_dict["private"],
        row_dict["#posts"], row_dict["#followers"], row_dict["#follows"],
        row_dict["followers_following_ratio"],
    ]])
    X = hstack([X_text, csr_matrix(X_num)]).tocsr()
    proba = rf.predict_proba(X)[0, 1]
    return proba, text


def main():
    st.title("🕵️ Fake Instagram Account Detector")
    st.caption("CCS3356 Natural Language Processing — Group 01, Neuralyx Labs")
    st.write(
        "Enter an Instagram profile's details below. The app applies the same "
        "NLP preprocessing and feature extraction used during training, then "
        "runs the best-performing model to classify the account as **Real** "
        "or **Fake**."
    )

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            profile_pic = st.selectbox("Has profile picture?", ["Yes", "No"])
            fullname_words = st.number_input("Words in full name", min_value=0, max_value=10, value=2)
            name_eq_username = st.selectbox("Full name equals username?", ["No", "Yes"])
            external_url = st.selectbox("External URL in bio?", ["No", "Yes"])
            private = st.selectbox("Private account?", ["No", "Yes"])
        with col2:
            username_digit_ratio = st.slider("Digit ratio in username", 0.0, 1.0, 0.0, 0.01)
            fullname_digit_ratio = st.slider("Digit ratio in full name", 0.0, 1.0, 0.0, 0.01)
            bio_length = st.number_input("Biography length (characters)", min_value=0, max_value=150, value=40)
            posts = st.number_input("Number of posts", min_value=0, value=20)
            followers = st.number_input("Number of followers", min_value=0, value=500)
            follows = st.number_input("Number of accounts followed", min_value=0, value=300)

        model_choice = st.radio(
            "Model to use", ["Logistic Regression (Himas)", "Random Forest (Afrith)"],
            horizontal=True,
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        follows_safe = max(follows, 1)
        row_dict = {
            "profile pic": 1 if profile_pic == "Yes" else 0,
            "nums/length username": username_digit_ratio,
            "fullname words": fullname_words,
            "nums/length fullname": fullname_digit_ratio,
            "name==username": 1 if name_eq_username == "Yes" else 0,
            "description length": bio_length,
            "external URL": 1 if external_url == "Yes" else 0,
            "private": 1 if private == "Yes" else 0,
            "#posts": posts,
            "#followers": followers,
            "#follows": follows,
            "followers_following_ratio": followers / follows_safe,
        }

        if model_choice.startswith("Logistic"):
            lr, vec, scaler = load_himas_model()
            if lr is None:
                st.error("Model files not found. Run `python -m src.member1_himas_lr_lstm` first.")
                return
            proba, verbalized_text = predict_logistic_regression(row_dict, lr, vec, scaler)
        else:
            rf, vec, scaler = load_afrith_model()
            if rf is None:
                st.error("Model files not found. Run `python -m src.member2_afrith_rf_rnn` first.")
                return
            proba, verbalized_text = predict_random_forest(row_dict, rf, vec, scaler)

        label = "Fake" if proba >= 0.5 else "Real"
        confidence = proba if label == "Fake" else 1 - proba

        st.divider()
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Uncertain — model confidence ({confidence:.0%}) is below the 70% threshold.")
        elif label == "Fake":
            st.error(f"🚩 Prediction: **FAKE** — confidence {confidence:.0%}")
        else:
            st.success(f"✅ Prediction: **REAL** — confidence {confidence:.0%}")

        with st.expander("Show engineered profile description used by the NLP model"):
            st.code(verbalized_text)

        st.caption(
            "Disclaimer: this prediction is a probabilistic estimate from a "
            "machine learning model and should not be used as the sole basis "
            "for flagging or removing an account. No profile data entered "
            "here is stored."
        )


if __name__ == "__main__":
    main()
