# STEP 1: Libraries Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# STEP 2: Dataset Load
data = pd.read_csv("train.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# STEP 3: Basic Info
print(data.info())
print(data.describe())

# STEP 4: Data Cleaning
data = data.fillna(0)  # Missing values fill

# Sirf numeric columns
data = data.select_dtypes(include=['int64', 'float64'])

# STEP 5: Features and Target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# STEP 6: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 7: Model Train
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 8: Prediction
predictions = model.predict(X_test)

# STEP 9: Error Check
error = mean_absolute_error(y_test, predictions)
print("Average Prediction Error:", error)

# STEP 10: Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()