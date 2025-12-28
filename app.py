
import streamlit as st
import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter house details below:")

sqft_living = st.number_input("Sqft Living", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0)
yr_renovated = st.selectbox("Renovated? (0 = No, 1 = Yes)", [0, 1])

if st.button("Predict Price"):
    X_new = np.array([[sqft_living, bedrooms, bathrooms, yr_renovated]])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)[0]
    st.success(f"Predicted House Price: ${round(prediction, 2)}")
