import pickle
import pandas as pd

# Load model
with open("models/house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Sample input (ONLY 10 FEATURES)
sample_house = pd.DataFrame([{
    "No of bedrooms": 4,
    "No of bathrooms": 3,
    "living area": 2500,
    "lot area": 5000,
    "No of floors": 2,
    "waterfront present": 0,
    "No of views": 2,
    "house condition": 4,
    "house grade": 8,
    "Built Year": 2010
}])

# Prediction
prediction = model.predict(sample_house)

print("🏠 Predicted Price:", prediction[0])