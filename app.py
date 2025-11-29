import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('mumbai_price_model.pkl')
columns = joblib.load('model_columns.pkl')

st.set_page_config(
    page_title="Mumbai Flat Price Predictor",
    page_icon="house",
    layout="centered"
)

st.title("Mumbai Flat Price Predictor")
st.markdown("Enter flat details below to get an instant price estimate")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Carpet Area (sq.ft)", min_value=200, max_value=10000, value=800, step=50)
    bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)

with col2:
    age = st.slider("Age of Property (years)", min_value=0, max_value=50, value=5)
    ready_to_move = st.checkbox("Ready to Move", value=True)
    parking = st.selectbox("Parking", options=["None", "Open", "Covered"], index=2)

if st.button("Predict Price", type="primary"):
    
    parking_score = 2 if parking == "Covered" else 1 if parking == "Open" else 0
    ready = 1 if ready_to_move else 0

    input_data = pd.DataFrame([[0] * len(columns)], columns=columns)

    input_data['carpet_area_sqft'] = area
    input_data['bhk'] = bhk
    input_data['bathrooms'] = bathrooms
    input_data['age_years'] = age
    input_data['ready_to_move'] = ready
    input_data['parking_score'] = parking_score
    input_data['facing_encoded'] = 4  

    log_price = model.predict(input_data)[0]
    predicted_price = np.expm1(log_price)

    crore = predicted_price / 10_000_000
    lakh = (predicted_price % 10_000_000) / 100_000

    price_text = f"₹{crore:.2f} Cr"

    st.success(f"### Predicted Price: **{price_text}**")
    st.info("Model Accuracy: 92.5%")

st.markdown("---")
st.caption("Random Forest Model : R² = 0.925")