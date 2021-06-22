#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training dataset
dataset = pd.read_csv('GOOGL.csv')
training_set = dataset.iloc[:-20,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60,4218):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

#Importing libraries for RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing RNN
regressor = Sequential()

#Adding first LSTM Layer and some Dropout regularisation
#1
regressor.add(LSTM(units = 100, return_sequences= True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding 3 more layers
#2
regressor.add(LSTM(units = 100, return_sequences= True))
regressor.add(Dropout(0.2))
#3
regressor.add(LSTM(units = 100, return_sequences= True))
regressor.add(Dropout(0.2))
#4
regressor.add(LSTM(units = 100, return_sequences= True))
regressor.add(Dropout(0.2))
#5
regressor.add(LSTM(units = 100, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))


#Adding Output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN model to the training set
regressor.fit(X_train, y_train, epochs=500, batch_size=32)


#Reading test dataset
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset.iloc[-20:,1:2].values

#Predicting stock price
#dataset_local = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset.iloc[len(training_set)-80:,1:2].values
inputs = inputs.reshape(-1,1)
inputs_scaled = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs_scaled[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.savefig('Pot.jpg',bbox_inches='tight',dpi=1000)
plt.show()

#Evaluation of the model
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)
