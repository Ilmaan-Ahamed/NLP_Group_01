# 🤖 Fake Social Media Account Detection

## 📌 Overview

This repository contains a complete **Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL)** project focused on detecting fake Instagram accounts.

The project aims to automatically classify Instagram profiles as **Real** or **Fake** by analyzing profile metadata and biography text using multiple Machine Learning and Deep Learning models. The best-performing model is integrated into a **Streamlit Web Application** for real-time prediction.

---

## 🎯 Project Objectives

* Detect fake Instagram accounts automatically.
* Apply NLP preprocessing techniques to biography text.
* Compare multiple Machine Learning and Deep Learning models.
* Evaluate model performance using industry-standard metrics.
* Deploy the best model through a user-friendly web application.
* Promote safer social media interactions and fraud prevention.

---

## 🧠 NLP Pipeline

### 📥 Data Collection

* Instagram Fake Spammer Genuine Accounts Dataset
* Data acquisition from Kaggle

### 🧹 Data Preprocessing

* Text Cleaning
* Lowercase Conversion
* URL Removal
* Emoji Removal
* Special Character Removal
* Missing Value Handling

### ✂️ Text Processing

* Tokenization
* Stop Word Removal
* Stemming
* Lemmatization

### 🔢 Feature Engineering

* Bag of Words (BoW)
* TF-IDF Vectorization
* BERT Embeddings

### 🤖 Model Training

* Machine Learning Models
* Deep Learning Models

### 📊 Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

---

## 🚀 Models Implemented

| Member         | Machine Learning Model | Deep Learning Model |
| -------------- | ---------------------- | ------------------- |
| 👨‍💻 Member 1 | Logistic Regression    | LSTM                |
| 👨‍💻 Member 2 | Random Forest          | CNN                 |
| 👨‍💻 Member 3 | SVM                    | BERT                |

---

## 📊 Evaluation Metrics

| Metric           | Purpose                                     |
| ---------------- | ------------------------------------------- |
| Accuracy         | Overall prediction correctness              |
| Precision        | Measures false positive reduction           |
| Recall           | Measures ability to detect fake accounts    |
| F1-Score         | Balance between Precision and Recall        |
| ROC-AUC          | Classification capability across thresholds |
| Confusion Matrix | Detailed prediction analysis                |

---

## 🌐 Web Application Features

### User Inputs

* Biography Text
* Number of Followers
* Number of Following
* Number of Posts
* Profile Picture Status
* Private/Public Account Status

### System Output

* Real Account ✅
* Fake Account ❌
* Confidence Score (%)

---

## 🛠 Technologies Used

### Programming Languages

* Python

### NLP Libraries

* NLTK
* spaCy
* Transformers (Hugging Face)

### Machine Learning

* Scikit-Learn

### Deep Learning

* TensorFlow
* Keras
* PyTorch

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
Fake-Social-Media-Account-Detection/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── processed/
│       ├── cleaned_train.csv
│       └── cleaned_test.csv
│
├── notebooks/
│   ├── member1_logreg_lstm.ipynb
│   ├── member2_rf_cnn.ipynb
│   ├── member3_svm_bert.ipynb
│   └── final_model_comparison.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_logreg.py
│   ├── train_lstm.py
│   ├── train_rf.py
│   ├── train_cnn.py
│   ├── train_svm.py
│   ├── train_bert.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── lstm_model.h5
│   ├── cnn_model.h5
│   ├── bert_model/
│   └── best_model.pkl
│
├── reports/
│   ├── Project_Validation_Report.pdf
│   ├── Final_Project_Report.pdf
│   ├── Model_Comparison_Report.pdf
│   └── Ethics_and_Bias_Analysis.pdf
│
├── screenshots/
│
├── videos/
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 System Workflow

```text
User Input
      │
      ▼
Text Preprocessing
      │
      ▼
Feature Extraction
(BoW / TF-IDF / BERT)
      │
      ▼
Trained Model
      │
      ▼
Prediction
      │
      ▼
Real ✅ / Fake ❌
+ Confidence Score
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/fake-social-media-account-detection.git
```

### 2️⃣ Navigate to the Project

```bash
cd fake-social-media-account-detection
```

### 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 4️⃣ Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Run Streamlit Application

```bash
streamlit run app/app.py
```

### 7️⃣ Open Browser

```text
http://localhost:8501
```

---

## 📈 Expected Outcomes

* Accurate fake account detection.
* Comparative analysis of ML and DL models.
* Real-world application of NLP techniques.
* Functional Streamlit deployment.
* Understanding of Responsible AI practices.

---

## ⚖️ Ethics & Responsible AI

### Potential Risks

* False Positives
* False Negatives
* Dataset Bias
* Platform-Specific Limitations

### Mitigation Strategies

* Cross Validation
* Threshold-Based Predictions
* Bias Analysis
* Human-in-the-Loop Verification
* Transparent Model Evaluation

---

## 🚀 Future Improvements

* Support Multiple Social Media Platforms
* Real-Time Account Analysis
* Explainable AI Dashboard
* Multi-Language Support
* Larger Dataset Training
* Cloud Deployment using AWS
* CI/CD Integration

---

## 👨‍💻 Contributors

| Name     | Role                       |
| -------- | -------------------------- |
| Member 1 | Logistic Regression & LSTM |
| Member 2 | Random Forest & CNN        |
| Member 3 | SVM & BERT                 |

---

## 🎓 Academic Information

**Module:** CCS3356 – Natural Language Processing

**Project Title:** Fake Social Media Account Detection Using Machine Learning and Deep Learning

**Institution:** SLTC Research University

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

Together, we can build safer social media platforms through AI and NLP. 🚀
