# 💰 Insurance Premium Prediction System

This project predicts the **medical insurance premium** amount for individuals based on their demographics, health conditions, and textual claim/feedback data. It uses **machine learning** and **natural language processing (NLP)** to analyze both structured and unstructured data.

The final model is deployed as a **Streamlit web application**.

🔗 **Live App**: [https://insurance-premium-predictiongit.streamlit.app/]

---

## 📦 Required Packages

To run this project, install the following Python packages:

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn nltk
Or install all dependencies using
pip install -r requirements.txt

```
## 📊 Dataset Overview

Source: Kaggle Insurance Premium Dataset

Rows: 1,338

Columns: 7 original + 3 generated text columns (claim_description, medical_notes, feedback)

Target: expenses (medical insurance premium)

Note: Text features were generated to enrich the dataset for NLP processing

## 🧼 Data Preprocessing

Checked and handled missing or duplicated rows

Encoded categorical features (sex, smoker, region)

Treated outliers in bmi and expenses using IQR capping

Scaled numerical features using StandardScaler

Applied text preprocessing for NLP:

Lowercasing

Removing punctuation/special characters

Tokenization

Stopword removal

Lemmatization

TF-IDF vectorization for text columns

## 🤖 Model Development

Models used:

Linear Regression, Lasso, Ridge

Decision Tree, Random Forest, Gradient Boosting, AdaBoost

KNN, XGBoost

Evaluation metrics:

MAE, RMSE, R² Score

5-fold cross-validation

## 🔍 Hyperparameter Tuning

Used RandomizedSearchCV to optimize:

Random Forest: n_estimators, max_depth, min_samples_split, max_features

XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

## 🚀 Deployment

Final model saved as insurance_pipeline.pkl

Pipeline includes:

Trained XGBoost model

StandardScaler for numerical features

TF-IDF vectorizers for claim_description, medical_notes, feedback

Streamlit app built for real-time prediction

Automatically preprocesses both structured and textual inputs

## 🧪 Run Locally
git clone https://github.com/Indrajaiswal/Insurance-Premium-Prediction.git




