from __future__ import division
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as  np


pd.options.mode.chained_assignment = None  # default='warn'   = turning off warning!!!

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

average = 0.0
for i in range(df.shape[0]):
    if i < 14:
        df['AverageTrueRange'] = 0
        average += float(df['TrueRange'][i])
        if i == 13:
            average = float(average / 14)
            df['AverageTrueRange'][i] = average
    #elif i >= 14:


#TODO calculate ADX @ https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
#TODO @ https://www.youtube.com/watch?v=LKDJQLrXedg

print df.head(40)
