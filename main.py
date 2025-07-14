import streamlit as st
import numpy as np
import pickle
import pandas as pd

rfc = pickle.load(open('model.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))


st.title('🏦 Bank Churn Prediction App')

# User inputs
credit_score = st.number_input('Credit Score', min_value=300, max_value=1000)
country = st.selectbox('Country', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100)
tenure = st.number_input('Tenure (Years)', min_value=0, max_value=10)
balance = st.number_input('Account Balance', min_value=0.0)
products_number = st.number_input('Number of Products', min_value=1, max_value=4)
credit_card = st.selectbox('Has Credit Card?', ['Yes', 'No'])
active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)


country_encoded = {'France': 0, 'Spain': 1, 'Germany': 2}[country]
gender_encoded = {'Male': 1, 'Female': 0}[gender]
credit_card_encoded = 1 if credit_card == 'Yes' else 0
active_member_encoded = 1 if active_member == 'Yes' else 0


if st.button('🔍 Predict'):
    try:
        features = np.array([[credit_score, country_encoded, gender_encoded, age, tenure, balance,
                              products_number, credit_card_encoded, active_member_encoded, estimated_salary]])



        pred = rfc.predict(features)[0]

        if pred == 1:
            st.error("❌ Customer is Left.")
        else:
            st.success("✅ Customer is likely to STAY.")
    except Exception as e:
        st.warning(f"⚠️ Something went wrong: {e}")
