#importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#Feature scaling
scaler = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scaler.fit_transform(training_set)

#Creating the timesteps
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

#Converting to numpy array for the neural network
X_train,y_train = np.array(X_train),np.array(y_train)

#Reshaping into the shape the neural network takes(samples,timestep,input_dimension)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#Initialising the Recurrent Network
recurrent_regressor = tf.keras.Sequential()
recurrent_regressor.add(tf.keras.layers.LSTM(units = 50,return_sequences = True,
                                             input_shape = (X_train.shape[1],1)))
recurrent_regressor.add(tf.keras.layers.Dropout(0.2))

#Second layer
recurrent_regressor.add(tf.keras.layers.LSTM(units = 50,return_sequences = True))
recurrent_regressor.add(tf.keras.layers.Dropout(0.2))

#Third layer
recurrent_regressor.add(tf.keras.layers.LSTM(units = 50,return_sequences = True))
recurrent_regressor.add(tf.keras.layers.Dropout(0.2))

#Fourth layer
recurrent_regressor.add(tf.keras.layers.LSTM(units = 50,return_sequences = False))
recurrent_regressor.add(tf.keras.layers.Dropout(0.2))

#Output layer
recurrent_regressor.add(tf.keras.layers.Dense(units = 1))

#Compiling the neural network
recurrent_regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

#Fitting the neural network
history = recurrent_regressor.fit(X_train,y_train,batch_size = 32,epochs = 100)

#Plotting the training
plt.plot(history.history['loss'],label = 'Loss')
plt.legend()
plt.show()

#Getting the test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)

#Reshaping into the shape the neural network takes(samples,timestep,input_dimension)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Predicting
predicted_stock_price = recurrent_regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#Visualising the resutl
plt.plot(real_stock_price,color = 'red',label = 'Real Google Stock')
plt.plot(predicted_stock_price,color = 'blue',label = 'Predicted Google Stock')
plt.legend()
plt.show()