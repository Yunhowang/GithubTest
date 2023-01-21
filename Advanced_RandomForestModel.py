#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('DataSet.csv',encoding='EUC-KR') #���ϴ� �Ⱓ�� �����͸� ��������

# select the relevant columns for the model
X = data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']] # choose relevant weather factors
y = data['������հ���']

# split the data into training and test sets
split_date = pd.Timestamp('01-06-2017')
date = data.loc[data['����']>'2017-06-01','����']
row = ['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�','������հ���']
train_data = data.loc[data['����']<'2017-06-01', row]
test_data = data.loc[data['����']>'2017-06-01',row]
X_train = train_data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']]
y_train = train_data[['������հ���']]
X_test = test_data[['��ձ��','�������','�ְ���','�ϰ�����','�ִ����ǳ��','�ִ�ǳ��','���ǳ��','����̽����µ�','�ּһ�����','��ջ�����','�����ð�','�������µ�','�����ʻ�µ�']]
y_test = test_data[['������հ���']]
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
