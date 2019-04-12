from __future__ import division
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from SwingTradeElite.SwingTradeElite.SwingTradeElite.Model.StockData import StockData

# Settings
pd.options.mode.chained_assignment = None  # default='warn'   = turning off warning!!!
style.use('ggplot')

# Ticker, start and end date - 6 months
ticker = 'TSLA'
start = dt.datetime(2018, 3, 23)
end = dt.datetime(2019, 3, 25)

# Create stockdata
stockdata = StockData(ticker, True, start, end)

stockdata.compare3data('MACD', 'MACDsmooth', 'Close')

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


