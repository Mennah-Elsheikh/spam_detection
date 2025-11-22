# 📧 Spam Detection using NLP

This project trains machine learning models to classify SMS messages as Ham (legitimate) or Spam (unwanted). It uses standard NLP preprocessing, TF-IDF features, and classifiers such as Multinomial Naive Bayes and Linear SVM. Trained models are saved to `models/` and example analyses are in the Jupyter notebook.

### 🌟 Project Highlights

- **Exploratory Data Analysis (EDA):** Class distribution checks and visualizations (e.g., word clouds).
- **Text Preprocessing:** Removal of URLs, numbers, punctuation, stopwords, and lemmatization.
- **N-Gram Example:** A simple Bigram Markov demonstration for sentence probability estimation.
- **Modeling:** TF-IDF vectorization, SMOTE for class imbalance, and training/evaluation of Naive Bayes and Linear SVM.

### Repository Structure

- `spam_detection.ipynb` : Main Jupyter notebook with preprocessing, EDA, modeling, and evaluation.
- `requirements.txt` : Python dependencies.
- `data/` : Place the dataset file here (e.g., `data/spam.csv`).
- `models/` : Trained model artifacts (pickles) are saved here by the notebook.

### 🛠️ Key Results (example SVM Model)

These are example results reported from the notebook (your run may vary depending on preprocessing and random seed):

| Metric              | Score  |
| :------------------ | :----- |
| **Accuracy**        | 0.939  |
| **F1-Score (Spam)** | 0.94   |

### 🚀 How to Run

1. Install dependencies (recommended in a virtual environment):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure the dataset is placed at `data/spam.csv`. If you need to download it manually, put it in that path.

3. Start Jupyter and open the notebook:

```powershell
jupyter notebook spam_detection.ipynb
```

4. Run the notebook cells sequentially. Trained model files will be saved to the `models/` folder.

### Loading a Saved Model (example)

You can load a saved model in Python like this:

```python
import joblib
model = joblib.load('models/svm_model.pkl')
```
