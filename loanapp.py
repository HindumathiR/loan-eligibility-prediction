import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_model.joblib")

st.title("🏦 Loan Eligibility Predictor")
st.markdown("Enter applicant details to check if loan will be approved.")

# Input widgets
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0, value=360)

# Predict button
if st.button("Check Loan Status"):
    row = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term
    }])
    
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    if pred == 1:
        st.success(f"✅ Loan Approved! (Probability: {proba:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Probability: {proba:.2f})")
