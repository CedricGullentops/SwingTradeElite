import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2018,3,23)
end = dt.datetime(2019,3,25)

#df = web.DataReader('TSLA','yahoo', start, end)
#df.to_csv('tsla.csv')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

dfnew = df['Adj Close']
print dfnew

#df['Adj Close'].plot()
#plt.show()

#df = df.drop(['DATE'], 1)

print df.shape[0] #rows
print df.shape[1] #cols

#print df.values #only values

print df.head(1)