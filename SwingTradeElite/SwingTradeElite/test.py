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

df = pd.read_csv('tsla.csv', parse_dates=['Date'], index_col=0)
#print df.dtypes #show types: High, Low, Open, Close, Volume, Adj Close

#df['Close'].plot(figsize=(16, 12))
#plt.show()

#df = df.drop(['DATE'], 1)

#print df.shape[0] #rows
#print df.shape[1] #cols

#print df.head(1)

