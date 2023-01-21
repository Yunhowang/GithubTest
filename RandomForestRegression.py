#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# load the data from the .csv file
data = pd.read_csv("DataSet.csv",encoding='EUC-KR')
from pandas.tseries.offsets import MonthEnd

data['����']=pd.to_datetime(data['����'])
scaler = MinMaxScaler()
data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']] = scaler.fit_transform(data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']])
scalery = MinMaxScaler()
data[['������հ���']] = scalery.fit_transform(data[['������հ���']])

# select the relevant columns for the model
X = data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']] # choose relevant weather factors
y = data['������հ���']

# split the data into training and test sets

date = data.loc[data['����']>'2017-06-01','����']
row = ['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�','������հ���']
train_data = data.loc[data['����']<'2017-05-01', row]
test_data = data.loc[data['����']>'2017-05-01',row]
X_train = train_data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']]
y_train = train_data[['������հ���']]
X_test = test_data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']]
y_test = test_data[['������հ���']]
X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = y_train.ravel()
# create the model
model = RandomForestRegressor()

# train the model
model.fit(X_train, y_train)


# make predictions
y_pred = model.predict(X_test)
y_pred = y_pred[:len(date)]

#draw the graph
plt.plot(date, y_test, label='Original Price')
plt.plot(date, y_pred, label='Predicted Price')
plt.legend()
plt.show()

y_pred = np.array(y_pred)
y_test = np.array(y_test)

# evaluate the accuarcy
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

errors = abs(y_pred - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# %%
