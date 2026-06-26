import streamlit as st
import pickle
import pandas as pd

# Load model
with open("models/house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🏠 House Price Prediction App")

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
living_area = st.number_input("Living Area", 500, 10000, 2000)
lot_area = st.number_input("Lot Area", 500, 20000, 5000)
floors = st.number_input("Floors", 1, 5, 1)

waterfront = st.selectbox("Waterfront", ["No", "Yes"])
views = st.number_input("Views", 0, 10, 0)

condition = st.selectbox("House Condition", [1, 2, 3, 4, 5])
grade = st.selectbox("House Grade", [1,2,3,4,5,6,7,8,9,10])

built_year = st.number_input("Built Year", 1900, 2026, 2010)

# Convert categorical
waterfront_val = 1 if waterfront == "Yes" else 0

# Predict button
if st.button("Predict Price"):
    
    input_data = pd.DataFrame([{
        "No of bedrooms": bedrooms,
        "No of bathrooms": bathrooms,
        "living area": living_area,
        "lot area": lot_area,
        "No of floors": floors,
        "waterfront present": waterfront_val,
        "No of views": views,
        "house condition": condition,
        "house grade": grade,
        "Built Year": built_year
    }])

    prediction = model.predict(input_data)

    st.success(f"🏠 Predicted Price: ₹ {prediction[0]:,.2f}")