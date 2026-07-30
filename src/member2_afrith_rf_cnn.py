"""
member2_afrith_rf_cnn.py
=========================
Member 02 - MM. Afrith | Git branch: AfrithNLP

Pipeline: Data Cleaning -> Lowercase -> Tokenization -> Stop-word Removal
          -> Stemming -> TF-IDF Vectorization
Models  : Random Forest (ML)  |  CNN (DL)

Usage:
    python -m src.member2_afrith_rf_cnn
"""

import os
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

from src.data_utils import load_split
from src.text_cleaning import preprocess_for_tfidf_cnn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_tfidf_features(train_text, test_text, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return vectorizer, X_train, X_test


def evaluate(y_true, y_pred, y_proba, model_name="Model"):
    print(f"\n===== {model_name} Evaluation =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    print("F1-Score :", f1_score(y_true, y_pred, zero_division=0))
    if y_proba is not None and len(set(y_true)) > 1:
        print("ROC-AUC  :", roc_auc_score(y_true, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))


def train_random_forest(X_train, y_train, X_val, y_val, feature_names=None):
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    evaluate(y_val, y_pred, y_proba, model_name="Random Forest (TF-IDF)")

    if feature_names is not None:
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("\nTop 15 most important features:")
        for i in top_idx:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")

    return clf


def build_cnn_model(vocab_size, max_len, embedding_dim=64):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(train_text, y_train, val_text, y_val, max_len=30, vocab_size=5000, epochs=10):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)

    train_seq = tokenizer.texts_to_sequences(train_text)
    val_seq = tokenizer.texts_to_sequences(val_text)

    train_pad = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
    val_pad = pad_sequences(val_seq, maxlen=max_len, padding="post", truncating="post")

    model = build_cnn_model(vocab_size, max_len)
    model.fit(
        train_pad, y_train,
        validation_data=(val_pad, y_val),
        epochs=epochs, batch_size=16, verbose=2,
    )

    y_proba = model.predict(val_pad).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    evaluate(y_val, y_pred, y_proba, model_name="CNN")

    return model, tokenizer


def main():
    numeric_df, text_series, y = load_split("train")

    print("Cleaning and preprocessing biography text (Afrith pipeline)...")
    cleaned_text = text_series.apply(preprocess_for_tfidf_cnn)

    idx_train, idx_val = train_test_split(
        numeric_df.index, test_size=0.2, random_state=42, stratify=y
    )

    num_train, num_val = numeric_df.loc[idx_train], numeric_df.loc[idx_val]
    text_train, text_val = cleaned_text.loc[idx_train], cleaned_text.loc[idx_val]
    y_train, y_val = y.loc[idx_train], y.loc[idx_val]

    # ---- Random Forest on TF-IDF + scaled numeric features ----
    vectorizer, tfidf_train, tfidf_val = build_tfidf_features(text_train, text_val)

    scaler = StandardScaler()
    num_train_scaled = scaler.fit_transform(num_train)
    num_val_scaled = scaler.transform(num_val)

    X_train_combined = hstack([tfidf_train, csr_matrix(num_train_scaled)]).tocsr()
    X_val_combined = hstack([tfidf_val, csr_matrix(num_val_scaled)]).tocsr()

    feature_names = list(vectorizer.get_feature_names_out()) + list(num_train.columns)

    rf_model = train_random_forest(X_train_combined, y_train, X_val_combined, y_val, feature_names)

    joblib.dump(rf_model, os.path.join(MODELS_DIR, "afrith_random_forest.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "afrith_tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "afrith_scaler.pkl"))

    # ---- CNN on raw (cleaned) text sequences ----
    try:
        cnn_model, tokenizer = train_cnn(text_train, y_train, text_val, y_val)
        cnn_model.save(os.path.join(MODELS_DIR, "afrith_cnn_model.keras"))
        joblib.dump(tokenizer, os.path.join(MODELS_DIR, "afrith_cnn_tokenizer.pkl"))
    except ImportError:
        print("TensorFlow not installed -- skipping CNN training. "
              "Install tensorflow to run the deep learning half of this pipeline.")

    print("\nMember 2 (Afrith) pipeline complete. Models saved to /models.")


if __name__ == "__main__":
    main()
