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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# FUNCTIONS (IMPORTANT: ALL FUNCTIONS TOP)
# ----------------------------

def explain_bmi(bmi):
    explanation = f"""
📌 BMI (Body Mass Index)  
Value: {bmi:.1f}

📊 Range Interpretation:
- 18.5–24.9 → Normal (low risk)
- 25.0–29.9 → Overweight (moderate risk)
- 30.0–34.9 → Obesity Class I (high risk)
- 35.0–39.9 → Obesity Class II (very high risk)
- ≥ 40 → Obesity Class III (extreme risk)
"""

    if bmi < 18.5:
        explanation += "\n👉 Underweight → Possible health risk"
    elif bmi < 25:
        explanation += "\n👉 Normal → Low insurance risk"
    elif bmi < 30:
        explanation += f"\n👉 Overweight → Slight premium increase (BMI {bmi:.1f})"
    elif bmi < 35:
        explanation += f"\n👉 Obesity Class I → Moderate risk (BMI {bmi:.1f})"
    elif bmi < 40:
        explanation += f"\n👉 Obesity Class II → High risk (BMI {bmi:.1f})"
    else:
        explanation += f"\n👉 Obesity Class III → Very high risk (BMI {bmi:.1f})"

    return explanation


def preprocess_text(text):
    if text is None or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ----------------------------
# LOAD MODEL
# ----------------------------
with open('insurance_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

model = pipeline['model']
scaler = pipeline['scaler']
tfidf_claim = pipeline['tfidf_claim']
tfidf_medical = pipeline['tfidf_medical']
tfidf_feedback = pipeline['tfidf_feedback']

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("🏥 Insurance Premium Prediction")

# ----------------------------
# INPUTS
# ----------------------------
st.sidebar.header("🧾 Customer Info")

age = st.sidebar.number_input("Age", 18, 100, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.number_input("Children", 0, 10, 0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

sex_val = 0 if sex == "Male" else 1
smoker_val = 1 if smoker == "Yes" else 0
region_val = {"northwest":0,"northeast":1,"southeast":2,"southwest":3}[region.lower()]

# ----------------------------
# TEXT INPUTS
# ----------------------------
claim_description = st.selectbox("Claim Description", [
    "Minor injury claim",
    "Moderate injury from car accident",
    "Severe car accident with multiple fractures"
])

medical_notes = st.selectbox("Medical Notes", [
    "No chronic illness",
    "Smoker, higher risk of lung issues",
    "Obese, potential risk for heart disease"
])

feedback = st.selectbox("Feedback", [
    "Quick claim, no issues",
    "Claim took long time but approved",
    "Very stressful experience"
])

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Insurance Expense"):

    claim_clean = preprocess_text(claim_description)
    medical_clean = preprocess_text(medical_notes)
    feedback_clean = preprocess_text(feedback)

    X_struct = scaler.transform([[age, bmi, children, sex_val, smoker_val, region_val]])

    X_claim = tfidf_claim.transform([claim_clean])
    X_medical = tfidf_medical.transform([medical_clean])
    X_feedback = tfidf_feedback.transform([feedback_clean])

    X = hstack([X_struct, X_claim, X_medical, X_feedback])

    pred_log = model.predict(X)
    pred = float(np.expm1(pred_log)[0])

    st.session_state["pred"] = pred
    st.session_state["age"] = age
    st.session_state["bmi"] = bmi
    st.session_state["children"] = children
    st.session_state["smoker"] = smoker
    st.session_state["claim"] = claim_description

# ----------------------------
# OUTPUT SECTION
# ----------------------------
if "pred" in st.session_state:

    st.success(f"💰 Predicted Expense: ${st.session_state['pred']:,.2f}")

    if st.button("🧠 Explain Prediction"):

        st.subheader("📊 Feature-wise Explanation")

        st.markdown("### 📌 Age")
        if st.session_state["age"] < 25:
            st.write("Low risk group")
        elif st.session_state["age"] < 45:
            st.write("Medium risk group")
        else:
            st.write("High risk group")

        st.markdown(explain_bmi(st.session_state["bmi"]))

        st.markdown("### 📌 Smoking")
        if st.session_state["smoker"] == "Yes":
            st.write("🚨 High risk factor (smoking increases premium)")
        else:
            st.write("✅ Low risk (non-smoker)")

        st.markdown("### 📌 Children")
        if st.session_state["children"] == 0:
            st.write("Low dependency → low impact")
        elif st.session_state["children"] <= 2:
            st.write("Moderate dependency")
        else:
            st.write("High dependency")

        st.markdown("### 📌 Claim Impact")
        if "accident" in st.session_state["claim"].lower():
            st.write("🚨 High cost claim type")
        else:
            st.write("🟢 Low cost claim")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("Developed by Indra Jaiswal ❤️")