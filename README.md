# 🛠🤖 Capstone Project: Spam Email Detection using Isolation Forest

This repository contains part of the capstone project from the course **Machine Learning (IT3190E, Semester 2024.2, HUST)**.  
It presents a complete pipeline for building an **Isolation Forest** model to tackle the problem of **spam email detection**.

---

## 📁 1. Dataset Preparation

- The project uses the **Enron Spam Dataset**, a well-known dataset derived from Enron Corporation's internal email archives.  
  ➤ Source: [Enron Spam Dataset on GitHub](https://github.com/MWiechmann/enron_spam_data)

- After cleaning and text normalization, the processed file `processed_data.csv` (located in `dataset/raw`) contains:
  - Original email content  
  - Corresponding labels (spam/ham)  
  - Cleaned versions of the text

- Training data was constructed with a **low proportion of anomalies** to align with the Isolation Forest assumption: anomalies are "few and different".  
  A separate testing set was built from the remaining samples.  
  ➤ Final datasets are available in `dataset/final/`.

---

## ⚙️ 2. Training Pipeline

- **TF-IDF Vectorization**: Emails were transformed into TF-IDF features using `TfidfVectorizer` from `scikit-learn`.  
  Hyperparameters like `min_df` and `max_df` were tuned to explore preprocessing effects.

- **Isolation Forest** (via `sklearn.ensemble.IsolationForest`):
  - `n_estimators = 100`
  - `max_samples = 256`

---

## 📊 3. Evaluation

- Model performance was evaluated using the **F1-score**, given the imbalanced nature of the dataset.
- Hyperparameters for TF-IDF (e.g., `min_df`, `max_df`) were optimized to maximize the F1-score.
