# Spam Detection Streamlit App
# Corrected version

# 1) Import Streamlit first
import streamlit as st

# 2) MUST be the first Streamlit command
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="centered"
)

# 3) Other imports
import re
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 4) Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
    except:
        pass

download_nltk_data()

# 5) Preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# 6) Load model + vectorizer
@st.cache_resource
def load_model():
    model_path = "models/best_spam_model.pkl"
    vectorizer_path = None
    for vec in ["binary_vectorizer.pkl", "tfidf_vectorizer.pkl", "count_vectorizer.pkl"]:
        if os.path.exists(f"models/{vec}"):
            vectorizer_path = f"models/{vec}"
            break

    if not os.path.exists(model_path):
        return None, None, "Model file not found!"
    if not vectorizer_path:
        return None, None, "Vectorizer file not found!"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer, None

def predict_spam(text, model, vectorizer):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    try:
        conf = max(model.predict_proba(features)[0]) * 100
    except:
        conf = None
    return pred, conf, cleaned

# 7) Main UI
st.title("📧 SMS Spam Detection")
st.markdown("### Enter an SMS message to check if it's spam or ham.")

model, vectorizer, error = load_model()
if error:
    st.error(f"⚠️ {error}")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

st.markdown("---")

# Ensure session state key exists
if "msg" not in st.session_state:
    st.session_state["msg"] = ""

# Text area bound to session_state
st.text_area(
    "Message",
    key="msg",
    height=150,
    placeholder="Type or paste an SMS message here...",
    label_visibility="collapsed"
)

# Example button callbacks
def load_ham():
    st.session_state["msg"] = "Hey! Are you free for dinner tonight? Let me know!"

def load_spam():
    st.session_state["msg"] = "URGENT! You have won $1000! Click here to claim your prize now!!!"

st.markdown("**Try example messages:**")
col1, col2 = st.columns(2)
with col1:
    st.button("📱 Ham Example", on_click=load_ham)
with col2:
    st.button("⚠️ Spam Example", on_click=load_spam)

# Predict button
if st.button("🔍 Classify Message", type="primary", use_container_width=True):
    text = st.session_state["msg"].strip()
    if text == "":
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            pred, conf, cleaned = predict_spam(text, model, vectorizer)

        st.markdown("---")
        st.markdown("### 📊 Result")
        if pred == 1:
            st.error("🚨 **SPAM detected!**")
        else:
            st.success("✅ **HAM (Not Spam)**")

        if conf is not None:
            st.metric("Confidence", f"{conf:.2f}%")
            st.progress(conf / 100)

        with st.expander("🔧 Preprocessing Details"):
            st.markdown("**Original:**")
            st.write(text)
            st.markdown("**Cleaned:**")
            st.write(cleaned or "(empty after cleaning)")

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.info(
        """
        This app uses a machine learning model trained on the
        SMS Spam Collection Dataset to classify messages.
        """
    )
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")
