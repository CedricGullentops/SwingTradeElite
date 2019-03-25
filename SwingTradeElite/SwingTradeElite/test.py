import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as  np

style.use('ggplot')

start = dt.datetime(2018,3,23)
end = dt.datetime(2019,3,25)

#df = web.DataReader('TSLA','yahoo', start, end)
#df.to_csv('tsla.csv')

df = pd.read_csv('tsla.csv', parse_dates=['Date'], index_col=0)
#print df.dtypes #show types: High, Low, Open, Close, Volume, Adj Close

#df['Close'].plot(figsize=(16, 12))
#plt.show()

df['TrueRange'] = df['High']-df['Low']
df['AverageTrueRange'] = np.nan

print df.head(5)

for i in range(13, df.shape[0]):
    df['AverageTrueRange'][i] = 10

print df.head(40)
