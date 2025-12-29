import streamlit as st
import joblib
import pandas as pd

# Load the frozen pipeline (Step 9)
model = joblib.load('credit_model.pkl')

st.title("Credit Risk Scoring Portal")
st.markdown("Enter customer details to assess loan default probability.")

# Create the Sidebar for User Input
st.sidebar.header("Customer Demographics")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
edu = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"])
income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=50000)

st.sidebar.header("Loan Details")
loan_amt = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=10000)
interest = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 11.0)
intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])

# Hidden/Static values for simulation (based on your columns)
# In a real app, these would also be inputs
st.sidebar.header("Credit History")
cred_hist = st.sidebar.number_input("Credit History Length (Years)", 0, 30, 5)
score = st.sidebar.number_input("Credit Score", 300, 850, 700)

# Process Data when button is clicked
if st.button("Calculate Risk"):
    # 1. Create DataFrame matching training structure
    input_df = pd.DataFrame({
        'person_age': [age], 'person_gender': [gender], 'person_education': [edu],
        'person_income': [income], 'person_emp_exp': [5], 'person_home_ownership': ['RENT'],
        'person_emp_length': [2.0], 'loan_intent': [intent], 'loan_grade': ['B'],
        'loan_amnt': [loan_amt], 'loan_int_rate': [interest], 'loan_percent_income': [loan_amt/income],
        'cb_person_default_on_file': ['N'], 'cb_person_cred_hist_length': [cred_hist],
        'previous_loan_defaults_on_file': ['No'], 'credit_score': [score]
    })

    # 2. Add Engineered Features (Step 4)
    input_df['loan_to_income_ratio'] = input_df['loan_amnt'] / input_df['person_income']
    input_df['int_burden'] = (input_df['loan_int_rate'] / 100) * input_df['loan_amnt']

    # 3. Predict using the RBF SVM Pipeline
    pred = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    # 4. Display Results
    if pred[0] == 1:
        st.error(f"Application REJECTED (Risk Probability: {prob*100:.2f}%)")
    else:
        st.success(f"Application APPROVED (Risk Probability: {prob*100:.2f}%)")