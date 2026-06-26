import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor


# Load dataset
data = pd.read_csv("data/house_data.csv")

print("===== First 5 Rows =====")
print(data.head())

print("\n===== Dataset Information =====")
print(data.info())

print("\n===== Missing Values =====")
print(data.isnull().sum())

print("\n===== Statistical Summary =====")
print(data.describe())



# Features
X = data[
    [
        "No of bedrooms",
        "No of bathrooms",
        "living area",
        "lot area",
        "No of floors",
        "waterfront present",
        "No of views",
        "house condition",
        "house grade",
        "Built Year"
    ]
]

y = data["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nFirst 5 Training Rows:")
print(X_train.head())

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




model = LinearRegression()


model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:5])




comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})
comparison["Predicted Price"] = comparison["Predicted Price"].round(2)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("\n===== Actual vs Predicted =====")
print(comparison.head(20))




comparison["Error"] = (
    comparison["Actual Price"]
    - comparison["Predicted Price"]
)

print(comparison.head(20))






mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print("\n===== Model Evaluation =====")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)



print(
    data.corr(numeric_only=True)["Price"]
    .sort_values(ascending=False)
)




# Create Model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_mse = mean_squared_error(y_test, rf_predictions)

rf_rmse = rf_mse ** 0.5

rf_r2 = r2_score(y_test, rf_predictions)

print("\n===== Random Forest Evaluation =====")
print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


print("\n===== Model Comparison =====")

print("Linear Regression R² :", r2)

print("Random Forest R²     :", rf_r2)




print("Train R²:", rf_model.score(X_train, y_train))
print("Test R² :", rf_model.score(X_test, y_test))


importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

print(
    importance_df.sort_values(
        by="Importance",
        ascending=False
    )
)



import pickle

with open("models/house_price_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Model saved successfully!")