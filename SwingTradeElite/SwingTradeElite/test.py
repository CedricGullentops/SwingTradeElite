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


def calculateIndicators(df):
    # Create new columns
    df['TrueRange'] = df['High'] - df['Low']
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
            if i == SMALLPERIOD - 1:
                average = average / SMALLPERIOD
                df['AverageTrueRange'][i] = average
        elif i >= SMALLPERIOD:
            df['AverageTrueRange'][i] = ((df['AverageTrueRange'][i - 1] * (SMALLPERIOD - 1)) + df['TrueRange'][
                i]) / SMALLPERIOD

        # Calculate High-PrevHigh and PrevLow-Low
        if i == 0:
            df['High-PrevHigh'] = 0.0
            df['PrevLow-Low'] = 0.0
        else:
            df['High-PrevHigh'][i] = df['High'][i] - df['High'][i - 1]
            df['PrevLow-Low'][i] = df['Low'][i - 1] - df['Low'][i]

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
            df['Smooth +DX'][i] = ((df['Smooth +DX'][i - 1] * (SMALLPERIOD - 1)) + df['+DX'][i]) / SMALLPERIOD
            df['Smooth -DX'][i] = ((df['Smooth -DX'][i - 1] * (SMALLPERIOD - 1)) + df['-DX'][i]) / SMALLPERIOD

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
            if i == MACDM - 1:
                df['EMA12'][i] = emamaverage / MACDM
        elif i >= MACDM:
            df['EMA12'][i] = df['Close'][i] * 2 / (MACDM + 1) + df['EMA12'][i - 1] * (1 - (2 / (MACDM + 1)))

        # Calculate 26 Day EMA
        if i < MACDL:
            df['EMA26'][i] = 0.0
            emalaverage += df['Close'][i]
            if i == MACDL - 1:
                df['EMA26'][i] = emalaverage / MACDL
        elif i >= MACDL:
            df['EMA26'][i] = df['Close'][i] * 2 / (MACDL + 1) + df['EMA26'][i - 1] * (1 - (2 / (MACDL + 1)))

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
            df['MACDsmooth'][i] = df['MACD'][i] * 2 / (MACDS + 1) + df['MACDsmooth'][i - 1] * (1 - (2 / (MACDS + 1)))

    # Delete the top SMALLPERIOD * 2 + 1 rows
    print("Removing top", 40, "rows")
    df = df.iloc[int(SMALLPERIOD * 3 + 1):]
    return df


def isTrending(dataframe):
    dataframe['IsTrending'] = np.nan
    for i in range(dataframe.shape[0]):
        if dataframe['ADX'][i] <= 20:
            dataframe['IsTrending'][i] = 0
        elif dataframe['ADX'][i] > 30:
            dataframe['IsTrending'][i] = 1
        else:
            if (dataframe['ADX'][i] - dataframe['ADX'][i-9]) > 0 and (dataframe['ADX'][i] - dataframe['ADX'][i-3]) > 0:
                dataframe['IsTrending'][i] = 1
            else:
                dataframe['IsTrending'][i] = 0
    return dataframe


def isPositiveTrend(dataframe):
    dataframe['IsPositiveTrend'] = np.nan
    for i in range(dataframe.shape[0]):
        if dataframe['IsTrending'][i] == 0:
            dataframe['IsPositiveTrend'][i] = 0
        elif (dataframe['+DMI'][i]- dataframe['-DMI'][i]) > 0:
            dataframe['IsPositiveTrend'][i] = 1
        else:
            dataframe['IsPositiveTrend'][i] = 0
    return dataframe


def isNegativeTrend(dataframe):
    dataframe['isNegativeTrend'] = np.nan
    for i in range(dataframe.shape[0]):
        if dataframe['IsTrending'][i] == 0:
            dataframe['isNegativeTrend'][i] = 0
        elif (dataframe['+DMI'][i]- dataframe['-DMI'][i]) < 0:
            dataframe['isNegativeTrend'][i] = 1
        else:
            dataframe['isNegativeTrend'][i] = 0
    return dataframe


def isMACDlarger(dataframe):
    dataframe['MACDlarger'] = np.nan
    for i in range(dataframe.shape[0]):
        if dataframe['MACD'][i] > dataframe['MACDsmooth'][i]:
            dataframe['MACDlarger'][i] = 1
        else:
            dataframe['MACDlarger'][i] = 0
    return dataframe


def isPositiveMACDCrossover(dataframe):
    dataframe['PosCrossover'] = np.nan
    for i in range(dataframe.shape[0]):
        if i != 0 and dataframe['MACD'][i] > dataframe['MACDsmooth'][i] and dataframe['MACD'][i-1] <= dataframe['MACDsmooth'][i-1]:
            dataframe['PosCrossover'][i] = 1
        else:
            dataframe['PosCrossover'][i] = 0
    return dataframe


def isNegativeMACDCrossover(dataframe):
    dataframe['NegCrossover'] = np.nan
    for i in range(dataframe.shape[0]):
        if i != 0 and dataframe['MACD'][i] <= dataframe['MACDsmooth'][i] and dataframe['MACD'][i-1] > dataframe['MACDsmooth'][i-1]:
            dataframe['NegCrossover'][i] = 1
        else:
            dataframe['NegCrossover'][i] = 0
    return dataframe


def setStopLosses(dataframe):
    dataframe['Stoploss'] = np.nan
    for i in range(3, dataframe.shape[0]):
        if dataframe['Close'][i] < dataframe['ADX'][i-1]:
            dataframe['Stoploss'][i] = dataframe['Stoploss'][i-1]
        else:
            if dataframe['ADX'][i] < dataframe['ADX'][i-3]:
                dataframe['Stoploss'][i] = 0.03 * dataframe['Close'][i]
            else:
                if dataframe['Close'][i] - dataframe['Close'][i-2] < 0.05 * dataframe['Close'][i]:
                    dataframe['Stoploss'][i] = dataframe['Close'][i] - dataframe['Close'][i-2]
                else:
                    dataframe['Stoploss'][i] = 0.05 * dataframe['Close'][i]
    return dataframe


def calulatePotentialProfits(dataframe):
    buy = []
    sell = []
    profit = []
    for i in range(dataframe.shape[0]):
        if dataframe['IsPositiveTrend'][i] == 1 and dataframe['MACDlarger'][i] == 1 and len(sell) == len(buy):
            buy.append(dataframe['Close'][i])
        if dataframe['Stoploss'][i] > dataframe['Close'][i] and len(buy) > len(sell):
            sell.append(dataframe['Close'][i])
        elif dataframe['NegCrossover'][i] == 1 and len(buy) > len(sell):
            sell.append(dataframe['Close'][i])
    for i in range(len(sell)):
        profit.append(sell[i]-buy[i])
    print(sell)
    print(buy)
    print(profit)
    return


def testProfits(dataframe):
    buy = []
    sell = []
    profit = []
    for i in range(dataframe.shape[0]):
        if dataframe['PosCrossover'][i] == 1 and len(sell) == len(buy):
            buy.append(dataframe['Close'][i])
        if dataframe['Stoploss'][i] > dataframe['Close'][i] and len(buy) > len(sell):
            sell.append(dataframe['Close'][i])
        elif dataframe['NegCrossover'][i] == 1 and len(buy) > len(sell):
            sell.append(dataframe['Close'][i])
    for i in range(len(sell)):
        profit.append(sell[i]-buy[i])
    print(sell)
    print(buy)
    print(profit)
    return


# Start and end date - 6 months
start = dt.datetime(2018, 3, 23)
end = dt.datetime(2019, 3, 25)

# downloaddata(start, end, 'SNPS')
downloaddata(start, end, 'SNPS')
df = readdata()
df = calculateIndicators(df)
df = isTrending(df)

# buy values
df = isPositiveTrend(df)
df = isMACDlarger(df)

# sell values
df = isNegativeMACDCrossover(df)

df = isPositiveMACDCrossover(df)

df = setStopLosses(df)

print("The way it's meant to be")
calulatePotentialProfits(df)

print("Purely positive crossover")
testProfits(df)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 4))  # 1 row, 2 columns
# df.plot(y=['MACD', 'MACDsmooth', 'IsPositiveTrend'], ax=ax1)
# df.plot(y=['Close'], ax=ax2)
# df.plot(y=['MACDlarger'], ax=ax3)
# plt.tight_layout()
# plt.show()


