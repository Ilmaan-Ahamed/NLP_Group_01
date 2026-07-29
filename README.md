# Fake Social Media Account Detection (Instagram)

**CCS3356 Natural Language Processing — Group Assignment**

---

## Project Title

**Fake Social Media Account Detection (Instagram)** — an NLP and machine
learning system that classifies Instagram profiles as **Real** or **Fake**
using profile-level features and text-based modeling techniques.

---

## Group Members

**Group No & Name:** 1 — Neuralyx Labs

| # | Student ID | Name | Git Branch |
|---|---|---|---|
| 1 | CIT–24–01–0369 | MJ. Ilmaan Ahamed (Team Lead) | `IlmaanNLP` |
| 2 | CIT–24–01–0297 | MM. Afrith | `AfrithNLP` |
| 3 | CIT–24–01–0302 | RM. Himas | `HimasNLP` |

**Git Repository:** https://github.com/Ilmaan-Ahamed/NLP_Group_01.git

---

## Problem Statement

Fake and bot Instagram accounts are widely used to spread misinformation,
conduct scams, manipulate engagement metrics, and harass genuine users.
Manually reviewing accounts at scale is impractical. This project builds
an automated NLP-based classification system that analyzes profile-level
signals — biography content/length, username patterns, follower and
following counts, post history, and account privacy — to determine whether
an Instagram account is **genuine** or **fraudulent**.

Each group member independently develops a different ML and DL model on
the same dataset; the best-performing model is integrated into a
Streamlit web application that outputs a **Real / Fake** prediction with a
confidence score.

**Intended users:** platform administrators, cybersecurity analysts,
digital marketers verifying audience authenticity, disinformation
researchers, and general users wanting to verify accounts they interact
with.

---

## Dataset Information

**Dataset:** Instagram Fake Spammer Genuine Accounts Dataset (Kaggle)
https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts

- **Size:** 5,000 profiles, perfectly balanced — 2,500 real (`fake = 0`), 2,500 fake (`fake = 1`)
- **Split used in this project:** 80/20 stratified → `data/train.csv` (4,000 rows), `data/test.csv` (1,000 rows)

| Column | Meaning |
|---|---|
| `profile pic` | 1 if the account has a profile picture, else 0 |
| `nums/length username` | ratio of digits to length in the username |
| `fullname words` | number of words in the full name |
| `nums/length fullname` | ratio of digits to length in the full name |
| `name==username` | 1 if full name equals username |
| `description length` | character length of the bio/description |
| `external URL` | 1 if the bio contains an external link |
| `private` | 1 if the account is private |
| `#posts` | number of posts |
| `#followers` | number of followers |
| `#follows` | number of accounts followed |
| `followers_following_ratio` | engineered ratio feature |
| `fake` | target label (0 = real, 1 = fake) |

**Note on biography text:** the dataset is fully structured/numeric and
does not include a raw bio string. To meet the project's NLP
feature-extraction requirement (BoW / TF-IDF / BERT), `src/data_utils.py`
verbalizes each row's structured signals into a short natural-language
profile description (e.g. *"no profile picture, empty biography, 12
followers, following 980 accounts, suspicious ratio"*). All three NLP
pipelines run on this verbalized text. If a version of the dataset with a
real `bio` column becomes available, the code will use it automatically
instead.

**Known limitations/biases:** privacy considerations around real public
profiles; possible regional/demographic underrepresentation; Instagram-
specific patterns that may not generalize to other platforms; short,
noisy bio text limiting feature richness.

---

## Setup Instructions

**Prerequisites:** Python 3.9–3.12, and internet access the first time you
run Member 3's BERT pipeline (downloads `bert-base-uncased`, then caches
it locally).

```bash
# 1. Move into the project folder (the one containing requirements.txt and src/)
cd project-root

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK resources
python -m src.text_cleaning --download
```

**Dependencies** (see `requirements.txt`): pandas, numpy, scikit-learn,
scipy, nltk, joblib, tensorflow (LSTM/CNN), torch + transformers (BERT),
streamlit (web app), and optionally matplotlib/seaborn/jupyter for
notebooks.

---

## How to Run the Project

```bash
# (Optional) Regenerate the train/test split from the raw dataset
python -m src.make_train_test_split

# Train each member's models (already-trained models are included in models/)
python -m src.member1_himas_lr_lstm          # Himas: Logistic Regression + LSTM
python -m src.member2_afrith_rf_cnn          # Afrith: Random Forest + CNN
python -m src.member3_ilmaan_svm_bert        # Ilmaan: SVM + fine-tuned BERT
python -m src.member3_ilmaan_svm_bert --skip-finetune   # SVM only, faster

# Compare all trained models (Accuracy, Precision, Recall, F1, ROC-AUC)
python -m src.compare_models

# Launch the Streamlit web application
streamlit run src/app.py
```

The app opens at `http://localhost:8501`. Enter a profile's details
(profile picture, bio length, followers, following, posts, private
status, etc.), choose a model, and click **Predict** to get a **Real /
Fake** label with a confidence score. Predictions below 70% confidence are
shown as **Uncertain**.

---

## Model Summary

| Member | Git Branch | Feature Extraction | ML Model | DL Model |
|---|---|---|---|---|
| Himas | `HimasNLP` | Bag-of-Words (BoW) | Logistic Regression | LSTM |
| Afrith | `AfrithNLP` | TF-IDF | Random Forest | CNN |
| Ilmaan | `IlmaanNLP` | BERT embeddings | SVM | Fine-tuned BERT (`bert-base-uncased`) |

**Shared preprocessing steps:** cleaning (remove URLs/emojis/punctuation),
tokenization, stop-word removal, lemmatization or stemming (member-
specific), then vectorization/embedding.

**Evaluation metrics used:** Accuracy, Precision, Recall, F1-Score,
Confusion Matrix, ROC-AUC — chosen together so no single metric hides a
model biased toward one class.

---

## Results Summary

Results from evaluating each trained model on the held-out test set
(`python -m src.compare_models`):

| Member | Model | Type | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Afrith | Random Forest | ML | 0.987 | 0.987 | 0.999 |
| Himas | Logistic Regression | ML | 0.967 | 0.967 | 0.995 |
| Afrith | CNN | DL | 0.936 | 0.934 | 0.984 |
| Himas | LSTM | DL | 0.917 | 0.912 | 0.981 |
| Ilmaan | SVM (BERT embeddings) | ML | *pending full run* | – | – |
| Ilmaan | Fine-tuned BERT | DL | *pending full run* | – | – |

**Current best model:** Random Forest (Afrith) — 98.7% accuracy, 99.9%
ROC-AUC. Its feature importances confirm that follower count, profile
picture presence, and post count are the strongest indicators of a fake
account.

**Next steps:** complete Member 3's BERT fine-tuning run, re-run
`compare_models.py` with all five models, and integrate the single
best-performing model into the final version of the Streamlit app.

---

## Folder Structure

```
project-root/
├── data/                # train.csv, test.csv, raw CSVs
├── notebooks/           # exploratory data analysis notebook
├── src/                 # all pipeline & app code
├── models/               # saved .pkl / .keras trained models
├── reports/              # generated evaluation reports (model_comparison.csv)
├── screenshots/          # repo & app screenshots for submission
├── videos/               # demo/progress video for submission
├── requirements.txt
├── README.md
└── .gitignore
```
