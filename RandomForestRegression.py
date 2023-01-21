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

data['일자']=pd.to_datetime(data['일자'])
scaler = MinMaxScaler()
data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']] = scaler.fit_transform(data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']])
scalery = MinMaxScaler()
data[['배추평균가격']] = scalery.fit_transform(data[['배추평균가격']])

# select the relevant columns for the model
X = data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']] # choose relevant weather factors
y = data['배추평균가격']

# split the data into training and test sets

date = data.loc[data['일자']>'2017-06-01','일자']
row = ['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도','배추평균가격']
train_data = data.loc[data['일자']<'2017-05-01', row]
test_data = data.loc[data['일자']>'2017-05-01',row]
X_train = train_data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']]
y_train = train_data[['배추평균가격']]
X_test = test_data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']]
y_test = test_data[['배추평균가격']]
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
