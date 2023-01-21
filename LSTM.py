#%%
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD

df = pd.read_csv("DataSet.csv",encoding='EUC-KR')
from pandas.tseries.offsets import MonthEnd

df['일자']=pd.to_datetime(df['일자'])

# Preprocess the data
# Clean and normalize the data
scaler = MinMaxScaler()
df[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']] = scaler.fit_transform(df[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']])
scalery = MinMaxScaler()
df[['배추평균가격']] = scalery.fit_transform(df[['배추평균가격']])

# select the relevant columns for the model
X = df[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']] # choose relevant weather factors
y = df['배추평균가격']

# split the data into training and test sets
split_date = pd.Timestamp('06-01-2017')
date = df.loc[df['일자']>'2017-06-01','일자']
row = ['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도','배추평균가격']
train_data = df.loc[df['일자']<'2017-06-01', row]
test_data = df.loc[df['일자']>'2017-06-01',row]
X_train = train_data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']]
y_train = train_data[['배추평균가격']]
X_test = test_data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']]
y_test = test_data[['배추평균가격']]

#reshape the Input
X_train = np.reshape(np.array(X_train), (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(np.array(X_test), (X_test.shape[0],X_test.shape[1],1))

# LSTM 아키텍처 (architecture)
my_LSTM_model = Sequential()
my_LSTM_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1), activation = 'tanh'))
my_LSTM_model.add(LSTM(units = 50, activation = 'tanh'))
my_LSTM_model.add(Dense(units=2))
# 컴파일링 (Compiling)
my_LSTM_model.compile(optimizer = SGD(learning_rate = 0.01, decay = 1e-7, momentum = 0.9, nesterov = False),loss = 'mean_squared_error')
# training data 세트에 피팅하기 (fitting)
my_LSTM_model.fit(X_train, y_train, epochs = 50, batch_size = 150, verbose = 0)

# X_test를 LSTM 모델에 넣어서 예측하기
LSTM_prediction = my_LSTM_model.predict(X_test)
LSTM_prediction = LSTM_prediction[:,0]
"""
y_test = np.array(y_test).reshape(-1,1)
LSTM_prediction = np.array(LSTM_prediction).reshape(-1,1)
LSTM_prediction = scalery.inverse_transform(LSTM_prediction)
y_test = scalery.inverse_transform(y_test)
"""
y_test=np.array(y_test)
LSTM_prediction = np.array(LSTM_prediction)

import matplotlib.pyplot as plt

# Plot the original prices and predicted prices
plt.plot(date,y_test, label='Original Prices')
plt.plot(date,LSTM_prediction, label='Predicted Prices')
plt.legend()
plt.show()

# evaluate the accuarcy
from sklearn.metrics import mean_absolute_error
print("MAE: ",mean_absolute_error(y_test, LSTM_prediction))

errors = abs(LSTM_prediction - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# %%
