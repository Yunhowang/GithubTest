#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('DataSet.csv',encoding='EUC-KR') #원하는 기간의 데이터를 넣으세요

# select the relevant columns for the model
X = data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']] # choose relevant weather factors
y = data['배추평균가격']

# split the data into training and test sets
split_date = pd.Timestamp('01-06-2017')
date = data.loc[data['일자']>'2017-06-01','일자']
row = ['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도','배추평균가격']
train_data = data.loc[data['일자']<'2017-06-01', row]
test_data = data.loc[data['일자']>'2017-06-01',row]
X_train = train_data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']]
y_train = train_data[['배추평균가격']]
X_test = test_data[['평균기온','최저기온','최고기온','일강수량','최대순간풍속','최대풍속','평균풍속','평균이슬점온도','최소상대습도','평균상대습도','가조시간','평균지면온도','최저초상온도']]
y_test = test_data[['배추평균가격']]
X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = y_train.ravel()


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
regressor.fit(X_train, y_train)
Advanced_RandomForest_prediction = regressor.predict(X_test)

#draw the graph
plt.plot(date, y_test, label='Original Price')
plt.plot(date, Advanced_RandomForest_prediction, label='Predicted Price')
plt.legend()
plt.show()

Advanced_RandomForest_prediction = np.array(Advanced_RandomForest_prediction)
y_test = np.array(y_test)

# evaluate the accuarcy
from sklearn.metrics import mean_absolute_error
print("MAE: ",mean_absolute_error(y_test, Advanced_RandomForest_prediction))

errors = abs(Advanced_RandomForest_prediction - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %%
