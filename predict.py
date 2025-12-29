import joblib
import pandas as pd

# 1. Load the saved model
model = joblib.load('credit_model.pkl')

# 2. Create a 'Fake' Customer with ALL required columns
# Make sure these values match the types in your loan_data.csv
new_customer = pd.DataFrame({
    'person_age': [25],
    'person_gender': ['male'],
    'person_education': ['Bachelor'],
    'person_income': [50000],
    'person_emp_exp': [3],
    'person_home_ownership': ['RENT'],
    'person_emp_length': [2.0],
    'loan_intent': ['PERSONAL'],
    'loan_grade': ['B'],
    'loan_amnt': [10000],
    'loan_int_rate': [11.0],
    'loan_percent_income': [0.2],
    'cb_person_default_on_file': ['N'],
    'cb_person_cred_hist_length': [4],
    'previous_loan_defaults_on_file': ['No'],
    'credit_score': [700]
})

# 3. Add the Engineered Features (Crucial Step!)
# The model expects these because we created them in app.py
new_customer['loan_to_income_ratio'] = new_customer['loan_amnt'] / new_customer['person_income']
new_customer['int_burden'] = (new_customer['loan_int_rate'] / 100) * new_customer['loan_amnt']

# 4. Make the Prediction
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print("\n--- Credit Risk Assessment ---")
if prediction[0] == 1:
    print("Result: REJECTED (High Risk of Default)")
else:
    print("Result: APPROVED (Low Risk)")

print(f"Confidence: {probability[0][prediction[0]]*100:.2f}%")