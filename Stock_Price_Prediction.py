#Stock Price Prediction (LSTM)
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
stock_data = yf.download('AAPL', start='2015-01-01', end='2020-01-01')
data = stock_data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
def create_dataset(data, time_step=60):
 X, y = [], []
 for i in range(len(data) - time_step - 1):
  X.append(data[i:(i + time_step), 0])
  y.append(data[i + time_step, 0])
  return np.array(X), np.array(y)
training_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:, :]
time_step = 60 
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Stock Price')
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='TrainingPredictions')
plt.plot(np.arange(len(scaled_data) - len(test_predict), len(scaled_data)), test_predict,
label='Test Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

plt.show()
