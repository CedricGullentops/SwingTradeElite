from __future__ import division
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as np

# Settings
pd.options.mode.chained_assignment = None  # default='warn'   = turning off warning!!!
style.use('ggplot')

# CONSTANTS
SMALLPERIOD = 14.0  # Period for Average True Range, Smooth +DX and Smooth -DX
MACDS = 9.0  # Period to smooth MACD
MACDM = 12.0  # Period for small EMA
MACDL = 26.0  # Period for big EMA


def downloaddata(startdate, enddate, stock):
    # Read data from yahoo and write to file
    dfdownload = web.DataReader(stock, 'yahoo', startdate, enddate)
    dfdownload.to_csv('download.csv')
    return


def readdata():
    # Read data from file and put it in a data frame
    return pd.read_csv('download.csv', parse_dates=['Date'], index_col=0)


def plotdata(dfplot, column):
    # Create and show a plot
    dfplot.plot(y=[column], figsize=(16, 12))
    plt.show()


def comparedata(dfplot, column, column2):
    # Create and show a plot
    dfplot.plot(y=[column, column2], figsize=(16, 12))
    plt.show()


def compare3data(dfplot, column, column2, column3):
    # Create and show a plot
    dfplot.plot(y=[column, column2, column3], figsize=(16, 12))
    plt.show()


def compare4data(dfplot, column, column2, column3, column4):
    # Create and show a plot
    dfplot.plot(y=[column, column2, column3, column4], figsize=(16, 12))
    plt.show()


# Start and end date - 6 months
start = dt.datetime(2018, 3, 23)
end = dt.datetime(2019, 3, 25)

# downloaddata(start, end, 'SNPS')
# downloaddata(start, end, 'TSLA')
df = readdata()

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
df['EMA12'] = np.nan
df['EMA26'] = np.nan
df['MACD'] = np.nan
df['MACDsmooth'] = np.nan

average = 0.0
smoothplusaverage = 0.0
smoothminaverage = 0.0
adxaverage = 0.0
emamaverage = 0.0
emalaverage = 0.0
MACDsmoothaverage = 0.0

print("Making calculations.")
for i in range(df.shape[0]):
    # Calculate AverageTrueRange
    if i < SMALLPERIOD:
        df['AverageTrueRange'] = 0.0
        average += df['TrueRange'][i]
        if i == SMALLPERIOD-1:
            average = average / SMALLPERIOD
            df['AverageTrueRange'][i] = average
    elif i >= SMALLPERIOD:
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
    if i < SMALLPERIOD + 1:
        df['Smooth +DX'][i] = 0.0
        df['Smooth -DX'][i] = 0.0
        smoothplusaverage += df['+DX'][i]
        smoothminaverage += df['-DX'][i]
        if i == SMALLPERIOD:
            smoothplusaverage = smoothplusaverage / SMALLPERIOD
            smoothminaverage = smoothminaverage / SMALLPERIOD
            df['Smooth +DX'][i] = smoothplusaverage
            df['Smooth -DX'][i] = smoothminaverage
    elif i >= SMALLPERIOD + 1:
        df['Smooth +DX'][i] = ((df['Smooth +DX'][i-1] * (SMALLPERIOD-1)) + df['+DX'][i]) / SMALLPERIOD
        df['Smooth -DX'][i] = ((df['Smooth -DX'][i-1] * (SMALLPERIOD-1)) + df['-DX'][i]) / SMALLPERIOD

    # Calculate +DMI and -DMI
    if i < SMALLPERIOD + 1:
        df['+DMI'][i] = 0.0
        df['-DMI'][i] = 0.0
    elif i >= SMALLPERIOD + 1:
        df['+DMI'][i] = (df['Smooth +DX'][i] / df['AverageTrueRange'][i]) * 100
        df['-DMI'][i] = (df['Smooth -DX'][i] / df['AverageTrueRange'][i]) * 100

    # Calculate DX
    if i < SMALLPERIOD + 1:
        df['DX'][i] = 0.0
        df['DX'][i] = 0.0
    elif i >= SMALLPERIOD + 1:
        df['DX'][i] = (abs(df['+DMI'][i] - df['-DMI'][i]) / (df['+DMI'][i] + df['-DMI'][i])) * 100

    # Calculate ADX
    if i < SMALLPERIOD + 1:
        df['ADX'][i] = 0.0
    elif SMALLPERIOD + 1 <= i < SMALLPERIOD * 2 + 1:
        df['ADX'][i] = 0.0
        adxaverage += df['DX'][i]
    elif i >= SMALLPERIOD * 2 + 1:
        df['ADX'][i] = ((df['ADX'][i - 1] * (SMALLPERIOD - 1)) + df['DX'][i]) / SMALLPERIOD

    # Calculate 12 Day EMA
    if i < MACDM:
        df['EMA12'][i] = 0.0
        emamaverage += df['Close'][i]
        if i == MACDM-1:
            df['EMA12'][i] = emamaverage / MACDM
    elif i >= MACDM:
        df['EMA12'][i] = df['Close'][i] * 2 / (MACDM + 1) + df['EMA12'][i-1] * (1 - (2 / (MACDM + 1)))

    # Calculate 26 Day EMA
    if i < MACDL:
        df['EMA26'][i] = 0.0
        emalaverage += df['Close'][i]
        if i == MACDL-1:
            df['EMA26'][i] = emalaverage / MACDL
    elif i >= MACDL:
        df['EMA26'][i] = df['Close'][i] * 2 / (MACDL + 1) + df['EMA26'][i-1] * (1 - (2 / (MACDL + 1)))

    # Calculate MACD
    df['MACD'][i] = df['EMA12'][i] - df['EMA26'][i]

    # Calculate MACD signal
    if i < MACDL:
        df['MACDsmooth'][i] = 0.0
    elif i < MACDL + MACDS:
        df['MACDsmooth'][i] = 0.0
        MACDsmoothaverage += df['MACD'][i]
        if i == MACDL + MACDS - 1:
            df['MACDsmooth'][i] = MACDsmoothaverage / MACDS
    elif i >= MACDL + MACDS:
        df['MACDsmooth'][i] = df['MACD'][i] * 2 / (MACDS + 1) + df['MACDsmooth'][i-1] * (1 - (2 / (MACDS + 1)))

# Delete the top SMALLPERIOD * 2 + 1 rows
print("Removing top", 40, "rows")
df = df.iloc[int(SMALLPERIOD * 3 + 1):]
print(df.head(40))

plotdata(df, 'Close')
plotdata(df, 'ADX')
comparedata(df, '+DMI', '-DMI')
comparedata(df, 'EMA12', 'EMA26')
comparedata(df, 'MACD', 'MACDsmooth')
