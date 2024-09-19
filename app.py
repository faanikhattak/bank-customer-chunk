import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load pre-trained model (replace with the path to your saved model)
model = load_model('my_model.h5')

# Load the preprocessor if used (for scaling, encoding, etc.)
# with open('preprocessor.pkl', 'rb') as f:
#     preprocessor = pickle.load(f)

# Define the app title
st.title("Bank Customer Churn Prediction")

# Collect user inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.number_input("Tenure", min_value=0, max_value=10, step=1)
balance = st.number_input("Balance", min_value=0.0, step=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0)
complain = st.selectbox("Complain", [0, 1])
satisfaction_score = st.number_input("Satisfaction Score", min_value=0, max_value=10, step=1)
point_earned = st.number_input("Points Earned", min_value=0, step=1)
geography = st.selectbox("Geography", ["Germany", "Spain", "Other"])
card_type = st.selectbox("Card Type", ["Gold", "Platinum", "Silver", "Other"])

# Encode categorical variables
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0
card_type_gold = 1 if card_type == "Gold" else 0
card_type_platinum = 1 if card_type == "Platinum" else 0
card_type_silver = 1 if card_type == "Silver" else 0

# Prepare input data for prediction
input_data = np.array([[credit_score, 1 if gender == "Male" else 0, age, tenure, balance, num_of_products,
                        has_cr_card, is_active_member, estimated_salary, complain, satisfaction_score,
                        point_earned, geography_germany, geography_spain, card_type_gold, 
                        card_type_platinum, card_type_silver]])

# Preprocess the data
#input_data = preprocessor.transform(input_data)

# Predict churn probability
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    churn_prob = prediction[0][0]
    
    if churn_prob > 0.5:
        st.error(f"The customer is likely to churn (Probability: {churn_prob:.2f})")
    else:
        st.success(f"The customer is unlikely to churn (Probability: {churn_prob:.2f})")
