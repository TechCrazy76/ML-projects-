# Fake News Detection Using TF-IDF and Logistic Regression | Natural Language Processing |
**Mar 2025**

**(Self project)**

◦ Built a **fake news classifier** using **TF-IDF** and **Logistic Regression** with text cleaning, stopword removal
and stemming; achieving **96.5%** test accuracy on *test set*, validated via **confusion matrix**, **precision recall**
and **F1-score** metrics

◦ Benchmarked multiple classifiers - **SVM (98.9%)** and **Random Forest (99.3%)**, and utilized **Scikit-learn
pipeline** for modular experimentation, preprocessing, and classification, ensuring robustness and minimal
misclassification

---

Welcome to the **Fake News Detector** — a machine learning project that classifies news articles as *real* or *fake* based on their textual content.  
This repository contains the complete implementation, from **data preprocessing and feature extraction** to **model training, evaluation, and benchmarking**.

---

## 🔍 Overview

Fake news has become a critical challenge in today’s information-driven world.  
This project leverages **Natural Language Processing (NLP)** and **Machine Learning** techniques to detect misleading or fabricated news articles automatically.  

The model combines **TF-IDF vectorization** and **Logistic Regression** to identify linguistic and contextual patterns that differentiate real and fake news, achieving a strong balance of interpretability and performance.

---

## ⚙️ Features

- **End-to-End Machine Learning Pipeline:** Covers dataset preparation, cleaning, feature extraction, training, and evaluation.  
- **Logistic Regression Classifier:** Core binary classification model for detecting fake news.  
- **Text Preprocessing:** Implements punctuation removal, tokenization, stopword elimination, and stemming using **NLTK**.  
- **TF-IDF Vectorization:** Converts text data into numerical representations reflecting word frequency and importance.  
- **Performance Evaluation:** Achieved **~96.5% test accuracy**, validated using **precision**, **recall**, **F1-score**, and **confusion matrix** metrics.  
- **Benchmarking:** Compared against other classifiers such as **SVM (98.9%)** and **Random Forest (99.3%)**, confirming model robustness.  

---

## 🧾 Dataset

The dataset used in this project originates from Kaggle’s [**Fake News Detection Competition**](https://www.kaggle.com/competitions/fake-news).  
Due to access limitations, a **public mirror dataset** with the same structure and labeling was used for implementation.  

It includes:
- **Columns:** `id`, `title`, `author`, `text`, `label`  
- **Labels:** `0` = Real news, `1` = Fake news  
- **Size:** ~20,000 training samples and ~5,000 test samples

---

## Dependencies

The project requires the following libraries and dependencies:

- NumPy
- Pandas
- scikit-learn
- nltk
- Jupyter Notebook

---

## 🧠 How It Works

1. **Data Preprocessing:**  
   - Combine `author`, `title`, and `text` fields into a single unified content column.  
   - Remove non-alphabetical characters, convert text to lowercase, remove stopwords, and apply **Porter Stemming**.  

2. **Feature Extraction:**  
   - Apply **TF-IDF Vectorization** using Scikit-learn to transform text into numerical form suitable for model input.  

3. **Model Training:**  
   - Train a **Logistic Regression** classifier on 80% of the data using stratified splitting to preserve class balance.  

4. **Evaluation:**  
   - Achieved **99% training accuracy** and **96.5% test accuracy**.  
   - Evaluated using **confusion matrix**, **precision**, **recall**, and **F1-score** to validate generalization.  

5. **Predictions:**  
   - The trained model can predict whether unseen articles are *fake* or *real* with high confidence.

---
