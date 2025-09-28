#Car Price Prediction (Neural Network)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
data = {
 'Age': [3, 4, 5, 2, 7, 3, 5, 6, 2, 1],
 'Mileage': [30000, 40000, 50000, 25000, 70000, 35000, 45000, 60000, 20000, 15000],
 'Horsepower': [150, 160, 170, 155, 180, 145, 175, 160, 155, 140],
 'Price': [20000, 18000, 16000, 22000, 14000, 21000, 17000, 15000, 23000, 24000]
}
df = pd.DataFrame(data)
X = df[['Age', 'Mileage', 'Horsepower']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
predictions = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})

print(predictions)
