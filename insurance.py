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
# NLTK setup
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
# PREDEFINED OPTIONS (DROPDOWN)
# ----------------------------

claim_options = [
    "Severe car accident with multiple fractures",
    "Major surgery following workplace injury",
    "Critical illness claim - cancer treatment",
    "Hospitalization due to severe infection",
    "Moderate injury from car accident",
    "Slip and fall - minor fracture",
    "Hospital stay for routine surgery",
    "Sports injury requiring short recovery",
    "Minor injury claim",
    "Doctor visit for checkup",
    "Outpatient treatment for flu",
    "Small medical claim for prescription"
]

medical_options = [
    "Smoker, higher risk of lung issues. Obese, potential risk for heart disease",
    "Overweight, moderate health risk. Diabetes under control",
    "Healthy weight range. No chronic illness",
    "History of hypertension. Asthma under observation",
    "No chronic illness"
]

feedback_options = [
    "Claim took long time but approved",
    "Satisfied with process despite delay",
    "Very stressful experience but finally resolved",
    "Happy with quick approval",
    "Process was smooth",
    "Good communication from company",
    "Quick claim, no issues",
    "Simple and efficient service",
    "Very satisfied with process"
]

# ----------------------------
# STREAMLIT DROPDOWN UI
# ----------------------------

st.markdown("## 📝 Claim Information")

claim_description = st.selectbox(
    "Select Claim Description",
    claim_options
)

medical_notes = st.selectbox(
    "Select Medical Notes",
    medical_options
)

feedback = st.selectbox(
    "Select Feedback",
    feedback_options
)
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
    pred_expense = np.expm1(pred_log)[0]

    st.session_state["pred_expense"] = float(pred_expense)
    st.session_state["inputs"] = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "claim": claim_description
    }

    # ----------------------------
# Show Result AFTER prediction
# ----------------------------
if "pred_expense" in st.session_state:
    pred_value = float(st.session_state["pred_expense"])

    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#00CC96,#009F6B);
        padding:25px;border-radius:15px;margin-top:20px'>
        <h2 style='color:white;text-align:center;'>💰 Predicted Insurance Expense</h2>
        <h1 style='color:white;text-align:center;font-size:50px'>
        ${pred_value:,.2f}</h1></h1>
        </div>
        """,
        unsafe_allow_html=True
    )

        # ----------------------------
    # EXPANDABLE EXPLANATION (2nd CLICK)
    # ----------------------------
    with st.expander("🧠 Click to understand WHY this prediction"):
        
        st.markdown("## 📊 Feature-wise Explanation")

        age = st.session_state["inputs"]["age"]
        bmi = st.session_state["inputs"]["bmi"]
        children = st.session_state["inputs"]["children"]
        smoker = st.session_state["inputs"]["smoker"]
        claim = st.session_state["inputs"]["claim"]

        # ---------------- AGE ----------------
        st.markdown("### 📌 Age (Full form: Age of Policy Holder)")
        if age < 25:
            st.write("🟢 Low risk group (18–25 years)")
        elif age < 45:
            st.write("🟡 Medium risk group (26–45 years)")
        else:
            st.write("🔴 High risk group (46+ years)")
        st.write(f"👉 Your age: {age}")

        # ---------------- BMI ----------------
        st.markdown("### 📌 BMI (Body Mass Index)")
        st.write("Formula: weight / height² (health risk indicator)")

        if bmi < 18.5:
            st.write("⚠️ Underweight → Moderate risk")
        elif 18.5 <= bmi < 25:
            st.write("✅ Normal range → Low risk")
        elif 25 <= bmi < 30:
            st.write("⚠️ Overweight → Increased risk")
        else:
            st.write("🚨 Obese → High risk (major premium increase)")

        st.write(f"👉 Your BMI: {bmi}")

        # ---------------- SMOKER ----------------
        st.markdown("### 📌 Smoking Status")
        if smoker == "Yes":
            st.write("🚨 Smoking increases premium by high margin (+ significant risk)")
        else:
            st.write("✅ Non-smoker → lower risk profile")

        # ---------------- CHILDREN ----------------
        st.markdown("### 📌 Number of Dependents (Children)")
        if children == 0:
            st.write("Low dependency → lower cost impact")
        elif children <= 2:
            st.write("Moderate dependency → slight increase")
        else:
            st.write("High dependency → higher insurance cost")

        # ---------------- CLAIM ----------------
        st.markdown("### 📌 Claim Description Impact")

        if "accident" in claim.lower() or "fracture" in claim.lower():
            st.write("🚨 Accident-related claim → HIGH cost impact")
            st.write("👉 This increases predicted expense significantly")
        elif "surgery" in claim.lower():
            st.write("⚠️ Surgery claim → Moderate to high cost impact")
        else:
            st.write("🟢 Minor claim → low cost impact")

        st.markdown("---")
        st.success("This explanation is generated using rule-based AI logic (feature interpretation layer).")
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



