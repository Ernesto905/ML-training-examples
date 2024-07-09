from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")

# Print the coefficients and intercept of the model
print("\nModel coefficients:")
for feature, coef in zip(california.feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Print some sample predictions
print("\nSample predictions:")
for i in range(5):
    true_value = y_test[i]
    predicted_value = y_pred[i]
    print(f"True value: {true_value:.2f}, Predicted value: {predicted_value:.2f}")
