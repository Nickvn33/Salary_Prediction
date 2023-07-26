import os
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

# write your code here
# print(data.head())
X = pd.DataFrame(data['rating'])
y = data['salary']
# print(X.head())
# print(y.head())
X = X ** 3
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
model = LinearRegression()
model.fit(X_train, y_train)
predictions_test = model.predict(X_test)
m = mape(y_test, predictions_test)
# print(f'{round(model.intercept_, 5)} {round(model.coef_[0], 5)} {round(m, 5)}')
print(round(m, 5))