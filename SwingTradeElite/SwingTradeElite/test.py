from __future__ import division
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as  np

# CONSTANTS
SMALLPERIOD = 14.0  # Period for Average True Range, Smooth +DX and Smooth -DX

pd.options.mode.chained_assignment = None  # default='warn'   = turning off warning!!!

style.use('ggplot')

# Start and end date - 6 months
start = dt.datetime(2018,3,23)
end = dt.datetime(2019,3,25)

# Read data from yahoo and write to file
# df = web.DataReader('TSLA','yahoo', start, end)
# df.to_csv('tsla.csv')

# Read data from file and put it in a data frame
df = pd.read_csv('tsla.csv', parse_dates=['Date'], index_col=0)
# print df.dtypes #show types: High, Low, Open, Close, Volume, Adj Close

# Create and show a plot
# df['Close'].plot(figsize=(16, 12))
# plt.show()

# Create new columns
df['TrueRange'] = df['High']-df['Low']
df['AverageTrueRange'] = np.nan
df['High-PrevHigh'] = np.nan
df['PrevLow-Low'] = np.nan
df['+DX'] = np.nan
df['-DX'] = np.nan
df['Smooth +DX'] = np.nan
df['Smooth -DX'] = np.nan
df['+DMI'] = np.nan
df['-DMI'] = np.nan
df['DX'] = np.nan
df['ADX'] = np.nan

average = 0.0
smoothplusaverage = 0.0
smoothminaverage = 0.0
adxaverage = 0.0

for i in range(df.shape[0]):
    # Calculate AverageTrueRange
    if i < 14:
        df['AverageTrueRange'] = 0.0
        average += df['TrueRange'][i]
        if i == 13:
            average = average / SMALLPERIOD
            df['AverageTrueRange'][i] = average
    elif i >= 14:
        df['AverageTrueRange'][i] = ((df['AverageTrueRange'][i-1] * (SMALLPERIOD-1)) + df['TrueRange'][i]) / SMALLPERIOD

    # Calculate High-PrevHigh and PrevLow-Low
    if i == 0:
        df['High-PrevHigh'] = 0.0
        df['PrevLow-Low'] = 0.0
    else:
        df['High-PrevHigh'][i] = df['High'][i] - df['High'][i-1]
        df['PrevLow-Low'][i] = df['Low'][i-1] - df['Low'][i]

    # Calculate +DX and -DX
    if i == 0:
        df['+DX'] = 0.0
        df['-DX'] = 0.0
    else:
        # +DX
        if df['High-PrevHigh'][i] > df['PrevLow-Low'][i] and df['High-PrevHigh'][i] > 0:
            df['+DX'][i] = df['High-PrevHigh'][i]
        else:
            df['+DX'][i] = 0.0

        # -DX
        if df['PrevLow-Low'][i] > df['High-PrevHigh'][i] and df['PrevLow-Low'][i] > 0:
            df['-DX'][i] = df['PrevLow-Low'][i]
        else:
            df['-DX'][i] = 0.0

    # Calculate Smooth +DX and Smooth -DX
    if i == 0:
        df['Smooth +DX'][i] = 0.0
        df['Smooth -DX'][i] = 0.0
    if i < 15:
        df['Smooth +DX'][i] = 0.0
        df['Smooth -DX'][i] = 0.0
        smoothplusaverage += df['+DX'][i]
        smoothminaverage += df['-DX'][i]
        if i == 14:
            smoothplusaverage = smoothplusaverage / SMALLPERIOD
            smoothminaverage = smoothminaverage / SMALLPERIOD
            df['Smooth +DX'][i] = smoothplusaverage
            df['Smooth -DX'][i] = smoothminaverage
    elif i >= 15:
        df['Smooth +DX'][i] = ((df['Smooth +DX'][i-1] * (SMALLPERIOD-1)) + df['+DX'][i]) / SMALLPERIOD
        df['Smooth -DX'][i] = ((df['Smooth -DX'][i-1] * (SMALLPERIOD-1)) + df['-DX'][i]) / SMALLPERIOD

    # Calculate +DMI and -DMI
    if i < 15:
        df['+DMI'][i] = 0.0
        df['-DMI'][i] = 0.0
    elif i >= 15:
        df['+DMI'][i] = (df['Smooth +DX'][i] / df['AverageTrueRange'][i]) * 100
        df['-DMI'][i] = (df['Smooth -DX'][i] / df['AverageTrueRange'][i]) * 100

    # Calculate DX
    if i < 15:
        df['DX'][i] = 0.0
        df['DX'][i] = 0.0
    elif i >= 15:
        df['DX'][i] = (abs(df['+DMI'][i] - df['-DMI'][i]) / (df['+DMI'][i] + df['-DMI'][i])) * 100

    # Calculate ADX
    if i < 15:
        df['ADX'][i] = 0.0
    elif 15 <= i < 29:
        df['ADX'][i] = 0.0
        adxaverage += df['DX'][i]
    elif i >= 29:
        df['ADX'][i] = ((df['ADX'][i - 1] * (SMALLPERIOD - 1)) + df['DX'][i]) / SMALLPERIOD

# TODO calculate ADX @ https://stockcharts.com/school/doku.php?id=chart_
#  school:technical_indicators:average_directional_index_adx

# TODO @ https://www.youtube.com/watch?v=LKDJQLrXedg

print(df.head(40))
