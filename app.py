import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("House Price Prediction")

sqft_living = st.number_input("Sqft Living", 100, 20000, 1000)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
yr_renovated = st.selectbox("Renovated?", [0, 1])

if st.button("Predict Price"):
    X_new = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    X_new["sqft_living"] = sqft_living
    X_new["bedrooms"] = bedrooms
    X_new["bathrooms"] = bathrooms
    X_new["yr_renovated"] = yr_renovated

    X_scaled = scaler.transform(X_new)
    prediction = model.predict(X_scaled)[0]

    st.success(f"Predicted Price: ${prediction:,.0f}")

