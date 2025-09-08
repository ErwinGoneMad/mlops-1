import joblib
import streamlit as st

regression = joblib.load("regression.joblib")

size = st.number_input("Size of the house", min_value=0)
num_rooms = st.number_input("Number of rooms", min_value=0)
garden = st.number_input("With or without a garden", min_value=0, max_value=1)

if st.button("Predict the price"):
    price = regression.predict([[size, num_rooms, garden]])
    st.write(f"The predicted price is {price[0]:.2f}€")
