# streamlit_insurance.py
import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# NLTK setup
# ----------------------------
# Download necessary NLTK data (first time only)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Load pipeline
# ----------------------------
with open('insurance_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

model = pipeline['model']
scaler = pipeline['scaler']
tfidf_claim = pipeline['tfidf_claim']
tfidf_medical = pipeline['tfidf_medical']
tfidf_feedback = pipeline['tfidf_feedback']

# ----------------------------
# Text preprocessing
# ----------------------------
def preprocess_text(text):
    if text is None or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💰",
    layout="wide"
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div style='background:linear-gradient(90deg, #1E90FF, #00CED1);padding:25px;border-radius:10px'>
    <h1 style='color:white;text-align:center;'>🏥 Insurance Premium Prediction</h1>
    <p style='color:white;text-align:center;font-size:18px;'>Predict customer insurance expense using structured & textual data</p>
    </div>
    """, unsafe_allow_html=True
)

# ----------------------------
# Sidebar: Structured Inputs
# ----------------------------
st.sidebar.header("🧾 Customer Information")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

sex_val = 0 if sex.lower() == "male" else 1
smoker_val = 1 if smoker.lower() == "yes" else 0
region_val = {"northwest":0, "northeast":1, "southeast":2, "southwest":3}[region.lower()]

# ----------------------------
# Info Cards Layout
# ----------------------------
st.markdown("### 🟢 Customer Details")
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div style='background-color:#FFD700;padding:20px;border-radius:10px;text-align:center'>"
              f"<h4>Age</h4><h2>{age}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div style='background-color:#FF7F50;padding:20px;border-radius:10px;text-align:center'>"
              f"<h4>BMI</h4><h2>{bmi}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div style='background-color:#20B2AA;padding:20px;border-radius:10px;text-align:center'>"
              f"<h4>Children</h4><h2>{children}</h2></div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
col4.markdown(f"<div style='background-color:#9370DB;padding:20px;border-radius:10px;text-align:center'>"
              f"<h4>Sex</h4><h2>{sex}</h2></div>", unsafe_allow_html=True)
col5.markdown(f"<div style='background-color:#FF69B4;padding:20px;border-radius:10px;text-align:center'>"
              f"<h4>Smoker</h4><h2>{smoker}</h2></div>", unsafe_allow_html=True)
col6.markdown(f"<div style='background-color:#40E0D0;padding:20px;border-radius:10px;text-align:center'>"
              f"<h4>Region</h4><h2>{region}</h2></div>", unsafe_allow_html=True)

# ----------------------------
# Main Page: Claim Info
# ----------------------------
st.markdown("<hr style='border:1px solid #D3D3D3'>", unsafe_allow_html=True)
st.markdown("<h2 style='color:#FF4500;'>📝 Claim Information</h2>", unsafe_allow_html=True)

claim_description = st.text_area("Claim Description", "Minor injury claim", height=100)
medical_notes = st.text_area("Medical Notes", "No chronic illness", height=100)
feedback = st.text_area("Customer Feedback", "Quick claim, no issues", height=100)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Insurance Expense", use_container_width=True):
    # Preprocess text
    claim_clean = preprocess_text(claim_description)
    medical_clean = preprocess_text(medical_notes)
    feedback_clean = preprocess_text(feedback)
    
    # Scale structured features
    structured_array = np.array([[age, bmi, children, sex_val, smoker_val, region_val]])
    structured_scaled = scaler.transform(structured_array)
    
    # Transform text features
    X_claim = tfidf_claim.transform([claim_clean])
    X_medical = tfidf_medical.transform([medical_clean])
    X_feedback = tfidf_feedback.transform([feedback_clean])
    
    # Combine all features
    X_input = hstack([structured_scaled, X_claim, X_medical, X_feedback])
    
    # Predict and reverse log transform
    pred_log = model.predict(X_input)
    pred_expense = np.expm1(pred_log)
    
    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#00CC96,#009F6B);padding:25px;border-radius:15px;margin-top:20px'>
        <h2 style='color:white;text-align:center;'>💰 Predicted Insurance Expense</h2>
        <h1 style='color:white;text-align:center;font-size:50px'>${pred_expense[0]:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """

    <div style='text-align:center;margin-top:50px;color:gray;'>
    Developed by Indra Jaiswal❤️
    </div>
    """, unsafe_allow_html=True
)



