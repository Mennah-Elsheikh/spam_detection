# SMS Spam Detection Project

## Overview
This project implements a comprehensive machine learning pipeline to detect spam SMS messages. It includes data preprocessing, feature extraction (Binary Encoding, Count Vectorization, TF-IDF, Word2Vec), model training (Naive Bayes, SVM, Random Forest), and a Streamlit web application for real-time predictions.

## Features
- **Data Preprocessing**: Text cleaning, tokenization, stopword removal, and lemmatization.
- **Feature Extraction**:
  - **Sentence-level**: Binary Encoding, Count Vectorization, TF-IDF.
  - **Word-level**: Word2Vec embeddings.
- **Models**: Naive Bayes, Support Vector Machine (SVM), and Random Forest.
- **Undersampling**: Handles class imbalance using `RandomUnderSampler`.
- **Deployment**: Interactive Streamlit web app.

## Project Structure
- `preprocessing_and_models.ipynb`: Main Jupyter notebook containing the complete pipeline (preprocessing, training, evaluation).
- `app.py`: Streamlit web application for deployment.
- `models/`: Directory containing saved models and vectorizers.
  - `best_spam_model.pkl`: The best performing model (SVM).
  - `binary_vectorizer.pkl`: Vectorizer for the best model.
- `requirements.txt`: List of Python dependencies.
- `EDA_and_ngrams.ipynb`: Notebook for Exploratory Data Analysis and N-gram analysis.
- `spam.csv`: Dataset file (in `data/` or root).

## Installation

1. Clone the repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
Run the Jupyter notebook `preprocessing_and_models.ipynb` to preprocess data, train models, and save the best one.
```bash
jupyter notebook preprocessing_and_models.ipynb
```
Execute all cells to generate `models/best_spam_model.pkl`.

### 2. Running the Web App
Launch the Streamlit app to test the model interactively:
```bash
streamlit run app.py
```
The app will open in your browser. You can enter SMS messages to check if they are classified as **Spam** or **Ham**.

## Results
The **SVM model with Binary Encoding** achieved the best performance:
- **F1 Score**: ~0.95
- **Accuracy**: ~98.6%

## Dependencies
- python
- pandas
- numpy
- scikit-learn
- nltk
- gensim
- imbalanced-learn
- matplotlib
- seaborn
- streamlit
