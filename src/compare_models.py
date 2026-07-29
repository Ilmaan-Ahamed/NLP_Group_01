"""
compare_models.py
==================
Section 4 - Model Comparison Plan

Loads every trained model from /models, evaluates all of them on the same
held-out test set (data/test.csv), and prints a comparison table using
Accuracy, Precision, Recall, F1-Score, and ROC-AUC so the group can pick
the best ML model and best DL model to integrate into the Streamlit app.

Usage:
    python -m src.compare_models
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.data_utils import load_split
from src.text_cleaning import (
    preprocess_for_bow_lstm, preprocess_for_tfidf_cnn, preprocess_for_bert,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def score(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan,
    }


def try_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def main():
    numeric_df, text_series, y = load_split("test")
    results = []

    # ---- Member 1: Logistic Regression (BoW) ----
    lr = try_load(os.path.join(MODELS_DIR, "himas_logistic_regression.pkl"))
    bow_vec = try_load(os.path.join(MODELS_DIR, "himas_bow_vectorizer.pkl"))
    himas_scaler = try_load(os.path.join(MODELS_DIR, "himas_scaler.pkl"))
    if lr and bow_vec and himas_scaler is not None:
        cleaned = text_series.apply(preprocess_for_bow_lstm)
        X_text = bow_vec.transform(cleaned)
        X_num = himas_scaler.transform(numeric_df)
        X = hstack([X_text, csr_matrix(X_num)])
        y_pred = lr.predict(X)
        y_proba = lr.predict_proba(X)[:, 1]
        results.append({"Member": "Himas", "Model": "Logistic Regression", "Type": "ML",
                         **score(y, y_pred, y_proba)})

    # ---- Member 2: Random Forest (TF-IDF) ----
    rf = try_load(os.path.join(MODELS_DIR, "afrith_random_forest.pkl"))
    tfidf_vec = try_load(os.path.join(MODELS_DIR, "afrith_tfidf_vectorizer.pkl"))
    afrith_scaler = try_load(os.path.join(MODELS_DIR, "afrith_scaler.pkl"))
    if rf and tfidf_vec and afrith_scaler is not None:
        cleaned = text_series.apply(preprocess_for_tfidf_cnn)
        X_text = tfidf_vec.transform(cleaned)
        X_num = afrith_scaler.transform(numeric_df)
        X = hstack([X_text, csr_matrix(X_num)]).tocsr()
        y_pred = rf.predict(X)
        y_proba = rf.predict_proba(X)[:, 1]
        results.append({"Member": "Afrith", "Model": "Random Forest", "Type": "ML",
                         **score(y, y_pred, y_proba)})

    # ---- Member 3: SVM (BERT embeddings) ----
    svm = try_load(os.path.join(MODELS_DIR, "ilmaan_svm.pkl"))
    ilmaan_scaler = try_load(os.path.join(MODELS_DIR, "ilmaan_scaler.pkl"))
    if svm and ilmaan_scaler is not None:
        try:
            from src.member3_ilmaan_svm_bert import extract_bert_embeddings
            cleaned = text_series.apply(preprocess_for_bert)
            embeddings = extract_bert_embeddings(cleaned)
            X_num = ilmaan_scaler.transform(numeric_df)
            X = np.hstack([embeddings, X_num])
            y_pred = svm.predict(X)
            y_proba = svm.predict_proba(X)[:, 1]
            results.append({"Member": "Ilmaan", "Model": "SVM (BERT emb.)", "Type": "ML",
                             **score(y, y_pred, y_proba)})
        except ImportError:
            print("torch/transformers not available -- skipping SVM/BERT evaluation.")
        except OSError as e:
            print(f"Could not download bert-base-uncased -- skipping SVM/BERT evaluation ({e}).")

    # ---- DL models (LSTM / CNN) ----
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        lstm_path = os.path.join(MODELS_DIR, "himas_lstm_model.keras")
        lstm_tok_path = os.path.join(MODELS_DIR, "himas_lstm_tokenizer.pkl")
        if os.path.exists(lstm_path) and os.path.exists(lstm_tok_path):
            model = load_model(lstm_path)
            tok = joblib.load(lstm_tok_path)
            cleaned = text_series.apply(preprocess_for_bow_lstm)
            seq = tok.texts_to_sequences(cleaned)
            pad = pad_sequences(seq, maxlen=30, padding="post", truncating="post")
            y_proba = model.predict(pad).ravel()
            y_pred = (y_proba >= 0.5).astype(int)
            results.append({"Member": "Himas", "Model": "LSTM", "Type": "DL",
                             **score(y, y_pred, y_proba)})

        cnn_path = os.path.join(MODELS_DIR, "afrith_cnn_model.keras")
        cnn_tok_path = os.path.join(MODELS_DIR, "afrith_cnn_tokenizer.pkl")
        if os.path.exists(cnn_path) and os.path.exists(cnn_tok_path):
            model = load_model(cnn_path)
            tok = joblib.load(cnn_tok_path)
            cleaned = text_series.apply(preprocess_for_tfidf_cnn)
            seq = tok.texts_to_sequences(cleaned)
            pad = pad_sequences(seq, maxlen=30, padding="post", truncating="post")
            y_proba = model.predict(pad).ravel()
            y_pred = (y_proba >= 0.5).astype(int)
            results.append({"Member": "Afrith", "Model": "CNN", "Type": "DL",
                             **score(y, y_pred, y_proba)})
    except ImportError:
        print("tensorflow not available -- skipping LSTM/CNN evaluation.")

    if not results:
        print("No trained models found in /models. Run each member's training "
              "script first (e.g. python -m src.member1_himas_lr_lstm).")
        return

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n===== Model Comparison (Section 4) =====")
    print(results_df.to_string(index=False))

    out_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to {out_path}")

    best_ml = results_df[results_df["Type"] == "ML"].head(1)
    best_dl = results_df[results_df["Type"] == "DL"].head(1)
    if not best_ml.empty:
        print(f"\nBest ML model: {best_ml.iloc[0]['Model']} ({best_ml.iloc[0]['Member']})")
    if not best_dl.empty:
        print(f"Best DL model: {best_dl.iloc[0]['Model']} ({best_dl.iloc[0]['Member']})")


if __name__ == "__main__":
    main()
