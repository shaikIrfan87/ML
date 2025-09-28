#House Price Prediction (Linear Regression)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
 'Size': [1500, 2000, 2500, 1800, 2300, 1200, 1600, 2200, 2100, 1700],
 'Bedrooms': [3, 4, 5, 3, 4, 2, 3, 4, 3, 2],
 'Price': [300000, 400000, 500000, 350000, 450000, 250000, 320000, 430000, 410000,
330000]
}
df = pd.DataFrame(data)
X = df[['Size', 'Bedrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
predictions = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})

print(predictions)
