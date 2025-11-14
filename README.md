# 📧 Spam Detection using NLP

This project aims to build and train machine learning models to classify SMS messages as either legitimate (Ham) or unwanted (Spam). It leverages Natural Language Processing (NLP) techniques for data preparation and uses models like Naive Bayes (NB) and Support Vector Machine (SVM) for classification.

### 🌟 Project Highlights

* **Exploratory Data Analysis (EDA):** Initial analysis of class distribution (Ham vs. Spam) and visualization using Word Clouds.
* **Advanced Text Preprocessing:** Includes steps like removal of URLs, numbers, punctuation, stopword removal, and Lemmatization.
* **N-Gram Markov Model:** Construction of a Bigram Markov Chain model to demonstrate sentence probability estimation.
* **Machine Learning Models:** Texts are converted using **TF-IDF Vectorization**, class imbalance is handled using **SMOTE**, and **Multinomial Naive Bayes** and **Linear Support Vector Machine (SVM)** models are trained and evaluated.

### 🛠️ Key Results (SVM Model)

The Linear SVM model, trained on TF-IDF features and SMOTE-resampled data, achieved strong performance:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.9785 |
| **F1-Score (Spam)** | 0.92 |

### 🚀 How to Run the Project

1.  **Dependencies:** Install the required libraries using the `requirements.txt` file.
2.  **Data:** Ensure the `spam.csv` dataset is available (the notebook typically handles downloading it from Kaggle).
3.  **Execution:** Run the cells sequentially in the `Spam_Detection_Enhanced_Structured.ipynb` notebook.