import pandas as pd
import numpy as np
import quandl, math, datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os.path

quandl_api_key = "myVxkYwXgrQqcdJKpyCC"
df = quandl.get('WIKI/GOOGL', api_key = quandl_api_key)

df = df[['Adj. Open','Adj. Close','Adj. High','Adj. Low','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'PCT_Change']]

df.fillna(-9999, inplace = True)

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-1 * forecast_out)

X = df.drop(['label'], axis = 1)
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# clf = LinearRegression()
# clf.fit(X_train, y_train)

##### Pickling #####
file_path = os.path.abspath(os.path.dirname(__file__))
pickle_path = os.path.join(file_path, "./resources/linearRegression.pickle")

# with open(pickle_path, 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open(pickle_path, 'rb')
clf = pickle.load(pickle_in)

##### END Pickling #####

##### Training and Prediction #####

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

##### END Training and Prediction #####

##### Ploting and Charting #####

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 12*60*60
next_unix = last_unix + one_day

for i in forecast_set:
    next_day = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_day] = [np.nan for _ in range(len(df.columns) -1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

##### END Ploting and Charting #####