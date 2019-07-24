from __future__ import division
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from SwingTradeElite.SwingTradeElite.SwingTradeElite.Model.StockData import StockData

# 4 Supervised Classification Learning Algorithms
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Settings
pd.options.mode.chained_assignment = None  # default='warn'   = turning off warning!!!
style.use('ggplot')

# Ticker, start and end date - 6 months
# Y M D
ticker = 'TSLA'
start = dt.datetime(2018, 3, 23)
end = dt.datetime(2019, 3, 25)

# Create stockdata
stockdata = StockData(ticker, True, start, end)

start = dt.datetime(2019, 3, 25)
end = dt.datetime(2019, 7, 24)
stockdata2 = StockData(ticker, True, start, end)

features2 = stockdata.dataframe.drop('NextWeekPrice', axis=1)
close2 = stockdata.dataframe['NextWeekPrice']

print(stockdata.dataframe['NextWeekPrice'])

stockdata.dataframe.sample(frac=1)

features = stockdata.dataframe.drop('NextWeekPrice', axis=1)
close = stockdata.dataframe['NextWeekPrice']

# X_train contains features for training, X_test contains features for testing
# test_size = 0.3 means 30% data for testing
# random_state = 1, is the seed value used by the random number generator
features_train, features_test, close_train, close_test = train_test_split(features, close, test_size=0.3, random_state=1)

clf_lr = LogisticRegression()
# fit the dataset into LogisticRegression Classifier
clf_lr.fit(features_train, close_train)
# predict on the unseen data
pred_lr = clf_lr.predict(features_test)

clf_knn = KNeighborsClassifier()
pred_knn = clf_knn.fit(features_train, close_train).predict(features_test) # method chainning

clf_rf = RandomForestClassifier(random_state=1)
pred_rf = clf_rf.fit(features_train, close_train).predict(features_test)

clf_dt = DecisionTreeClassifier()
pred_dt = clf_dt.fit(features_train, close_train).predict(features_test)


print("Accuracy of Logistic Regression:", accuracy_score(pred_lr, close_test))

print("Accuracy of KNN:", accuracy_score(pred_knn, close_test))

print("Accuracy of Random Forest:", accuracy_score(pred_rf, close_test))

print("Accuracy of Decision Tree:", accuracy_score(pred_dt, close_test))

print(classification_report(pred_lr, close_test))

print(clf_rf.predict(features2))

print(close2)

counter = 0.0
for i in range(stockdata2.dataframe.shape[0]):
    if stockdata2.dataframe['NextWeekPrice'][i] == close2[i]:
        counter += 1.0
print("score = %.2f" % (counter/stockdata2.dataframe.shape[0]))

#stockdata.compare3data('MACD', False, 'MACDsmooth', False, 'Close', True)

#print("The way it's meant to be")
#calulatePotentialProfits(df)

# print("Purely positive crossover")
# testProfits(df)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 4))  # 1 row, 2 columns
# df.plot(y=['MACD', 'MACDsmooth'], ax=ax1)
# df.plot(y=['Close'], ax=ax2)
# df.plot(y=['MACDlarger', 'IsPositiveTrend'], ax=ax3)
# plt.tight_layout()
# plt.show()


