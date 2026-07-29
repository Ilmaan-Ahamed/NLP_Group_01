"""
member3_ilmaan_svm_bert.py
===========================
Member 03 - MJ. Ilmaan Ahamed | Git branch: IlmaanNLP

Pipeline: Dataset Exploration -> Text Cleaning -> BERT WordPiece Tokenization
          -> Lemmatization -> BERT Embedding Extraction
Models  : SVM on BERT embeddings (ML)  |  Fine-tuned BERT classifier (DL)

Usage:
    python -m src.member3_ilmaan_svm_bert
    python -m src.member3_ilmaan_svm_bert --skip-finetune   # SVM only, faster
"""

import os
import argparse
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

from src.data_utils import load_split
from src.text_cleaning import preprocess_for_bert

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BERT_MODEL_NAME = "bert-base-uncased"


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


def extract_bert_embeddings(texts, batch_size=16, max_len=64):
    """
    Extract [CLS]-token sentence embeddings from bert-base-uncased for a
    list of cleaned biography strings.
    """
    import torch
    from transformers import BertTokenizer, BertModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    texts = list(texts)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            cls_embeddings = output.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def train_svm(X_train, y_train, X_val, y_val):
    clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    evaluate(y_val, y_pred, y_proba, model_name="SVM (BERT embeddings)")
    return clf


def finetune_bert_classifier(train_texts, y_train, val_texts, y_val, epochs=3, batch_size=16, max_len=64):
    """
    Fine-tune bert-base-uncased end-to-end for binary classification using
    HuggingFace's Trainer API.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        Trainer, TrainingArguments,
    )

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    class BioDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                list(texts), truncation=True, padding=True, max_length=max_len
            )
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = BioDataset(train_texts, y_train)
    val_dataset = BioDataset(val_texts, y_val)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "bert_finetune_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\n===== Fine-tuned BERT Evaluation =====")
    print(metrics)

    save_dir = os.path.join(MODELS_DIR, "ilmaan_bert_finetuned")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-finetune", action="store_true",
                         help="Skip the (slower) end-to-end BERT fine-tuning step")
    args = parser.parse_args()

    numeric_df, text_series, y = load_split("train")

    print("Cleaning biography text (Ilmaan pipeline)...")
    cleaned_text = text_series.apply(preprocess_for_bert)

    idx_train, idx_val = train_test_split(
        numeric_df.index, test_size=0.2, random_state=42, stratify=y
    )

    num_train, num_val = numeric_df.loc[idx_train], numeric_df.loc[idx_val]
    text_train, text_val = cleaned_text.loc[idx_train], cleaned_text.loc[idx_val]
    y_train, y_val = y.loc[idx_train], y.loc[idx_val]

    try:
        print("Extracting BERT embeddings for training set...")
        train_embeddings = extract_bert_embeddings(text_train)
        print("Extracting BERT embeddings for validation set...")
        val_embeddings = extract_bert_embeddings(text_val)
    except ImportError:
        print("transformers/torch not installed -- cannot run BERT pipeline. "
              "Install torch + transformers to run Member 3's pipeline.")
        return
    except OSError as e:
        print(
            "Could not download 'bert-base-uncased' from huggingface.co. "
            "This requires an internet connection with access to "
            "huggingface.co (the model is downloaded once and then cached "
            f"locally). Original error: {e}"
        )
        return

    scaler = StandardScaler()
    num_train_scaled = scaler.fit_transform(num_train)
    num_val_scaled = scaler.transform(num_val)

    X_train_combined = np.hstack([train_embeddings, num_train_scaled])
    X_val_combined = np.hstack([val_embeddings, num_val_scaled])

    svm_model = train_svm(X_train_combined, y_train, X_val_combined, y_val)

    joblib.dump(svm_model, os.path.join(MODELS_DIR, "ilmaan_svm.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "ilmaan_scaler.pkl"))

    if not args.skip_finetune:
        finetune_bert_classifier(text_train, y_train, text_val, y_val)
    else:
        print("Skipping BERT fine-tuning (--skip-finetune passed).")

    print("\nMember 3 (Ilmaan) pipeline complete. Models saved to /models.")


if __name__ == "__main__":
    main()
