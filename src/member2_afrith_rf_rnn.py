"""
member2_afrith_rf_rnn.py
=========================
Member 02 - MM. Afrith | Git branch: AfrithNLP

Pipeline: Data Cleaning -> Lowercase -> Tokenization -> Stop-word Removal
          -> Stemming -> TF-IDF Vectorization
Models  : Random Forest (ML)  |  RNN (DL)

Why RNN (per lecturer's guidance): a Simple Recurrent Neural Network
processes the profile description token-by-token, maintaining a hidden
state that carries information from earlier tokens forward. This lets it
capture sequential/order-dependent patterns in the text, unlike TF-IDF or
Bag-of-Words, which treat the text as an unordered bag of tokens. Compared
to the LSTM used in Member 1's pipeline, a SimpleRNN has no gating
mechanism, so it is more prone to vanishing gradients on longer sequences
-- but on the short (10-15 token) verbalized profile descriptions used
here, that weakness is less pronounced, and it offers a useful, simpler
baseline to compare against Member 1's LSTM and Member 3's BERT.

Usage:
    python -m src.member2_afrith_rf_rnn
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
from src.text_cleaning import preprocess_for_tfidf_rnn

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


def build_rnn_model(vocab_size, max_len, embedding_dim=64):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_rnn(train_text, y_train, val_text, y_val, max_len=30, vocab_size=5000, epochs=10):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)

    train_seq = tokenizer.texts_to_sequences(train_text)
    val_seq = tokenizer.texts_to_sequences(val_text)

    train_pad = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
    val_pad = pad_sequences(val_seq, maxlen=max_len, padding="post", truncating="post")

    model = build_rnn_model(vocab_size, max_len)
    model.fit(
        train_pad, y_train,
        validation_data=(val_pad, y_val),
        epochs=epochs, batch_size=16, verbose=2,
    )

    y_proba = model.predict(val_pad).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    evaluate(y_val, y_pred, y_proba, model_name="RNN")

    return model, tokenizer


def main():
    numeric_df, text_series, y = load_split("train")

    print("Cleaning and preprocessing biography text (Afrith pipeline)...")
    cleaned_text = text_series.apply(preprocess_for_tfidf_rnn)

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

    # ---- RNN on raw (cleaned) text sequences ----
    try:
        rnn_model, tokenizer = train_rnn(text_train, y_train, text_val, y_val)
        rnn_model.save(os.path.join(MODELS_DIR, "afrith_rnn_model.keras"))
        joblib.dump(tokenizer, os.path.join(MODELS_DIR, "afrith_rnn_tokenizer.pkl"))
    except ImportError:
        print("TensorFlow not installed -- skipping RNN training. "
              "Install tensorflow to run the deep learning half of this pipeline.")

    print("\nMember 2 (Afrith) pipeline complete. Models saved to /models.")


if __name__ == "__main__":
    main()