#%%
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD

df = pd.read_csv("DataSet.csv",encoding='EUC-KR')
from pandas.tseries.offsets import MonthEnd

df['����']=pd.to_datetime(df['����'])

# Preprocess the data
# Clean and normalize the data
scaler = MinMaxScaler()
df[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']] = scaler.fit_transform(df[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']])
scalery = MinMaxScaler()
df[['������հ���']] = scalery.fit_transform(df[['������հ���']])

# select the relevant columns for the model
X = df[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']] # choose relevant weather factors
y = df['������հ���']

# split the data into training and test sets
split_date = pd.Timestamp('01-01-2017')
date = df.loc[df['����']>'2017-06-01','����']
row = ['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�','������հ���']
train_data = df.loc[df['����']<'2017-06-01', row]
test_data = df.loc[df['����']>'2017-06-01',row]
X_train = train_data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']]
y_train = train_data[['������հ���']]
X_test = test_data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']]
y_test = test_data[['������հ���']]

#reshape the Input
X_train = np.reshape(np.array(X_train), (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(np.array(X_test), (X_test.shape[0],X_test.shape[1],1))

# GRU ��Ű��ó (architecture )
my_GRU_model = Sequential()
my_GRU_model.add(GRU(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1), activation = 'tanh'))
my_GRU_model.add(GRU(units = 50, activation = 'tanh'))
my_GRU_model.add(Dense(units = 2))

# �����ϸ� (Compiling)
my_GRU_model.compile(optimizer = SGD(learning_rate = 0.01, decay = 1e-7, momentum = 0.9, nesterov = False), loss = 'mean_squared_error')
    
# �����ϱ� (Fitting)
my_GRU_model.fit(X_train, y_train, epochs = 50, batch_size = 150, verbose = 0)
    
GRU_prediction = my_GRU_model.predict(X_test)
GRU_prediction = GRU_prediction[:,0]
"""
y_test = np.array(y_test).reshape(-1,1)
GRU_prediction = np.array(GRU_prediction).reshape(-1,1)
GRU_prediction = scalery.inverse_transform(GRU_prediction)
y_test = scalery.inverse_transform(y_test)
"""
y_test = np.array(y_test)
GRU_prediction = np.array(GRU_prediction)

import matplotlib.pyplot as plt

# Plot the original prices and predicted prices
plt.plot(date,y_test, label='Original Prices')
plt.plot(date,GRU_prediction, label='Predicted Prices')
plt.legend()
plt.show()

GRU_prediction = np.array(GRU_prediction)
y_test = np.array(y_test)

# evaluate the accuarcy
from sklearn.metrics import mean_absolute_error
print("MAE: ",round(mean_absolute_error(y_test, GRU_prediction)),2)

errors = abs(GRU_prediction - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# %%
