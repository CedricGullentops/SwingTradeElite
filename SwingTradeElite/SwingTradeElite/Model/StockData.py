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


class StockData:
    # CONSTANTS
    SMALLPERIOD = 14.0  # Period for Average True Range, Smooth +DX and Smooth -DX
    MACDS = 9.0  # Period to smooth MACD
    MACDM = 12.0  # Period for small EMA
    MACDL = 26.0  # Period for big EMA
    dataframe = None

    def __init__(self, ticker, isDownload, startdate, enddate):
        self.ticker = ticker
        if isDownload:
            self.downloaddata(startdate, enddate)
        self.readdata()
        self.dataframe = self.calculateIndicators(self.dataframe)
        self.dataframe = self.isTrending(self.dataframe)

        # buy values
        self.dataframe = self.isPositiveTrend(self.dataframe)
        self.dataframe = self.isMACDlarger(self.dataframe)

        # sell values
        self.dataframe = self.isNegativeMACDCrossover(self.dataframe)
        #self.dataframe = self.isPositiveMACDCrossover(self.dataframe)
        #self.dataframe = self.setStopLosses(self.dataframe)

    def downloaddata(self, startdate, enddate):
        # Read data from yahoo and write to file
        dfdownload = web.DataReader(self.ticker, 'yahoo', startdate, enddate)
        dfdownload.to_csv('download.csv')
        return

    def readdata(self):
        # Read data from file and put it in a data frame
        self.dataframe = pd.read_csv('download.csv', parse_dates=['Date'], index_col=0)
        return

    def plotdata(self, column, scale):
        # Create and show a plot
        if scale:
            self.dataframe['Temporary'] = self.dataframe[column]
            sum = 0
            for i in range(self.dataframe.shape[0]):
               sum += self.dataframe[column][i]
            sum /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i]-sum
            self.dataframe.plot(y=['Temporary'], figsize=(16, 12))
        else:
            self.dataframe.plot(y=[column], figsize=(16, 12))
        plt.show()

    def comparedata(self, column, scale, column2, scale2):
        # Create and show a plot
        if scale and not scale2:
            self.dataframe['Temporary'] = self.dataframe[column]
            sum = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column][i]
            sum /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i] - sum
            self.dataframe.plot(y=['Temporary', column2], figsize=(16, 12))

        elif scale2 and not scale:
            self.dataframe['Temporary'] = self.dataframe[column]
            sum = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column2][i]
            sum /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column2][i] - sum
            self.dataframe.plot(y=[column, 'Temporary'], figsize=(16, 12))

        elif scale and scale2:
            self.dataframe['Temporary'] = self.dataframe[column]
            self.dataframe['Temporary2'] = self.dataframe[column2]
            sum = 0
            sum2 = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column][i]
                sum2 += self.dataframe[column2][i]
            sum /= self.dataframe.shape[0]
            sum2 /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i] - sum
                self.dataframe['Temporary2'][i] = self.dataframe[column2][i] - sum2
            self.dataframe.plot(y=['Temporary', 'Temporary2'], figsize=(16, 12))

        else:
            self.dataframe.plot(y=[column, column2], figsize=(16, 12))
        plt.show()

    def compare3data(self, column, scale, column2, scale2, column3, scale3):
        # Create and show a plot
        if scale and not scale2 and not scale3:
            self.dataframe['Temporary'] = self.dataframe[column]
            sum = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column][i]
            sum /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i] - sum
            self.dataframe.plot(y=['Temporary', column2, column3], figsize=(16, 12))

        elif scale2 and not scale and not scale3:
            self.dataframe['Temporary'] = self.dataframe[column]
            sum = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column2][i]
            sum /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column2][i] - sum
            self.dataframe.plot(y=[column, 'Temporary', column3], figsize=(16, 12))

        elif scale and scale2 and not scale3:
            self.dataframe['Temporary'] = self.dataframe[column]
            self.dataframe['Temporary2'] = self.dataframe[column2]
            sum = 0
            sum2 = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column][i]
                sum2 += self.dataframe[column2][i]
            sum /= self.dataframe.shape[0]
            sum2 /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i] - sum
                self.dataframe['Temporary2'][i] = self.dataframe[column2][i] - sum2
            self.dataframe.plot(y=['Temporary', 'Temporary2', column3], figsize=(16, 12))

        elif scale and not scale2 and scale3:
            self.dataframe['Temporary'] = self.dataframe[column]
            self.dataframe['Temporary2'] = self.dataframe[column]
            sum = 0
            sum2 = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column][i]
                sum2 += self.dataframe[column3][i]
            sum /= self.dataframe.shape[0]
            sum2 /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i] - sum
                self.dataframe['Temporary2'][i] = self.dataframe[column3][i] - sum2
            self.dataframe.plot(y=['Temporary', column2, 'Temporary2'], figsize=(16, 12))

        elif scale2 and not scale and scale3:
            self.dataframe['Temporary'] = self.dataframe[column]
            self.dataframe['Temporary2'] = self.dataframe[column]
            sum = 0
            sum2 = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column2][i]
                sum2 += self.dataframe[column3][i]
            sum /= self.dataframe.shape[0]
            sum2 /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column2][i] - sum
                self.dataframe['Temporary2'][i] = self.dataframe[column3][i] - sum2
            self.dataframe.plot(y=[column, 'Temporary', 'Temporary2'], figsize=(16, 12))

        elif scale and scale2 and scale3:
            self.dataframe['Temporary'] = self.dataframe[column]
            self.dataframe['Temporary2'] = self.dataframe[column2]
            self.dataframe['Temporary3'] = self.dataframe[column3]
            sum = 0
            sum2 = 0
            sum3 = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column][i]
                sum2 += self.dataframe[column2][i]
                sum3 += self.dataframe[column3][i]
            sum /= self.dataframe.shape[0]
            sum2 /= self.dataframe.shape[0]
            sum3 /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column][i] - sum
                self.dataframe['Temporary2'][i] = self.dataframe[column2][i] - sum2
                self.dataframe['Temporary3'][i] = self.dataframe[column3][i] - sum3
            self.dataframe.plot(y=['Temporary', 'Temporary2', 'Temporary3'], figsize=(16, 12))

        elif not scale and not scale2 and scale3:
            self.dataframe['Temporary'] = self.dataframe[column3]
            sum = 0
            for i in range(self.dataframe.shape[0]):
                sum += self.dataframe[column3][i]
            sum /= self.dataframe.shape[0]
            for i in range(self.dataframe.shape[0]):
                self.dataframe['Temporary'][i] = self.dataframe[column3][i] - sum
            self.dataframe.plot(y=[column, column2, 'Temporary'], figsize=(16, 12))

        else:
            self.dataframe.plot(y=[column, column2, column3], figsize=(16, 12))
        plt.show()

    def compare4data(self, column, column2, column3, column4):
        # Create and show a plot
        # Scaling not implemented because of 16 possible variations
        self.dataframe.plot(y=[column, column2, column3, column4], figsize=(16, 12))
        plt.show()

    def calculateIndicators(self, df):
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
        df['NextWeekPrice'] = np.nan

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
            if i < self.SMALLPERIOD:
                df['AverageTrueRange'] = 0.0
                average += df['TrueRange'][i]
                if i == self.SMALLPERIOD - 1:
                    average = average / self.SMALLPERIOD
                    df['AverageTrueRange'][i] = average
            elif i >= self.SMALLPERIOD:
                df['AverageTrueRange'][i] = ((df['AverageTrueRange'][i - 1] * (self.SMALLPERIOD - 1)) + df['TrueRange'][
                    i]) / self.SMALLPERIOD

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
            if i < self.SMALLPERIOD + 1:
                df['Smooth +DX'][i] = 0.0
                df['Smooth -DX'][i] = 0.0
                smoothplusaverage += df['+DX'][i]
                smoothminaverage += df['-DX'][i]
                if i == self.SMALLPERIOD:
                    smoothplusaverage = smoothplusaverage / self.SMALLPERIOD
                    smoothminaverage = smoothminaverage / self.SMALLPERIOD
                    df['Smooth +DX'][i] = smoothplusaverage
                    df['Smooth -DX'][i] = smoothminaverage
            elif i >= self.SMALLPERIOD + 1:
                df['Smooth +DX'][i] = ((df['Smooth +DX'][i - 1] * (self.SMALLPERIOD - 1)) + df['+DX'][i]) / self.SMALLPERIOD
                df['Smooth -DX'][i] = ((df['Smooth -DX'][i - 1] * (self.SMALLPERIOD - 1)) + df['-DX'][i]) / self.SMALLPERIOD

            # Calculate +DMI and -DMI
            if i < self.SMALLPERIOD + 1:
                df['+DMI'][i] = 0.0
                df['-DMI'][i] = 0.0
            elif i >= self.SMALLPERIOD + 1:
                df['+DMI'][i] = (df['Smooth +DX'][i] / df['AverageTrueRange'][i]) * 100
                df['-DMI'][i] = (df['Smooth -DX'][i] / df['AverageTrueRange'][i]) * 100

            # Calculate DX
            if i < self.SMALLPERIOD + 1:
                df['DX'][i] = 0.0
                df['DX'][i] = 0.0
            elif i >= self.SMALLPERIOD + 1:
                df['DX'][i] = (abs(df['+DMI'][i] - df['-DMI'][i]) / (df['+DMI'][i] + df['-DMI'][i])) * 100

            # Calculate ADX
            if i < self.SMALLPERIOD + 1:
                df['ADX'][i] = 0.0
            elif self.SMALLPERIOD + 1 <= i < self.SMALLPERIOD * 2 + 1:
                df['ADX'][i] = 0.0
                adxaverage += df['DX'][i]
            elif i >= self.SMALLPERIOD * 2 + 1:
                df['ADX'][i] = ((df['ADX'][i - 1] * (self.SMALLPERIOD - 1)) + df['DX'][i]) / self.SMALLPERIOD

            # Calculate 12 Day EMA
            if i < self.MACDM:
                df['EMA12'][i] = 0.0
                emamaverage += df['Close'][i]
                if i == self.MACDM - 1:
                    df['EMA12'][i] = emamaverage / self.MACDM
            elif i >= self.MACDM:
                df['EMA12'][i] = df['Close'][i] * 2 / (self.MACDM + 1) + df['EMA12'][i - 1] * (1 - (2 / (self.MACDM + 1)))

            # Calculate 26 Day EMA
            if i < self.MACDL:
                df['EMA26'][i] = 0.0
                emalaverage += df['Close'][i]
                if i == self.MACDL - 1:
                    df['EMA26'][i] = emalaverage / self.MACDL
            elif i >= self.MACDL:
                df['EMA26'][i] = df['Close'][i] * 2 / (self.MACDL + 1) + df['EMA26'][i - 1] * (1 - (2 / (self.MACDL + 1)))

            # Calculate MACD
            df['MACD'][i] = df['EMA12'][i] - df['EMA26'][i]

            # Calculate MACD signal
            if i < self.MACDL:
                df['MACDsmooth'][i] = 0.0
            elif i < self.MACDL + self.MACDS:
                df['MACDsmooth'][i] = 0.0
                MACDsmoothaverage += df['MACD'][i]
                if i == self.MACDL + self.MACDS - 1:
                    df['MACDsmooth'][i] = MACDsmoothaverage / self.MACDS
            elif i >= self.MACDL + self.MACDS:
                df['MACDsmooth'][i] = df['MACD'][i] * 2 / (self.MACDS + 1) + df['MACDsmooth'][i - 1] * (1 - (2 / (self.MACDS + 1)))

        # Delete the top SMALLPERIOD * 2 + 1 rows
        df = df.iloc[int(self.SMALLPERIOD * 3 + 1):]

        for i in range(df.shape[0]-7):
            if df['Close'][i+7] > df['Close'][i]:
                df['NextWeekPrice'][i] = 1
            else:
                df['NextWeekPrice'][i] = 0
        df.drop(df.tail(7).index, inplace=True)
        return df

    @staticmethod
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

    @staticmethod
    def isPositiveTrend(dataframe):
        dataframe['IsPositiveTrend'] = np.nan
        for i in range(dataframe.shape[0]):
            if dataframe['IsTrending'][i] == 0:
                dataframe['IsPositiveTrend'][i] = 0
            elif (dataframe['+DMI'][i] - dataframe['-DMI'][i]) > 0:
                dataframe['IsPositiveTrend'][i] = 1
            else:
                dataframe['IsPositiveTrend'][i] = 0
        return dataframe

    @staticmethod
    def isNegativeTrend(dataframe):
        dataframe['isNegativeTrend'] = np.nan
        for i in range(dataframe.shape[0]):
            if dataframe['IsTrending'][i] == 0:
                dataframe['isNegativeTrend'][i] = 0
            elif (dataframe['+DMI'][i] - dataframe['-DMI'][i]) < 0:
                dataframe['isNegativeTrend'][i] = 1
            else:
                dataframe['isNegativeTrend'][i] = 0
        return dataframe

    @staticmethod
    def isMACDlarger(dataframe):
        dataframe['MACDlarger'] = np.nan
        for i in range(dataframe.shape[0]):
            if dataframe['MACD'][i] > dataframe['MACDsmooth'][i]:
                dataframe['MACDlarger'][i] = 1
            else:
                dataframe['MACDlarger'][i] = 0
        return dataframe

    @staticmethod
    def isPositiveMACDCrossover(dataframe):
        dataframe['PosCrossover'] = np.nan
        for i in range(dataframe.shape[0]):
            if i != 0 and dataframe['MACD'][i] > dataframe['MACDsmooth'][i] and dataframe['MACD'][i-1] <= dataframe['MACDsmooth'][i-1]:
                dataframe['PosCrossover'][i] = 1
            else:
                dataframe['PosCrossover'][i] = 0
        return dataframe

    @staticmethod
    def isNegativeMACDCrossover(dataframe):
        dataframe['NegCrossover'] = np.nan
        for i in range(dataframe.shape[0]):
            if i != 0 and dataframe['MACD'][i] <= dataframe['MACDsmooth'][i] and dataframe['MACD'][i-1] > dataframe['MACDsmooth'][i-1]:
                dataframe['NegCrossover'][i] = 1
            else:
                dataframe['NegCrossover'][i] = 0
        return dataframe

    @staticmethod
    def setStopLosses(dataframe):
        dataframe['Stoploss'] = np.nan
        for i in range(3, dataframe.shape[0]):
            if dataframe['ADX'][i] < dataframe['ADX'][i-1]:
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

    @staticmethod
    def setStopLoss(dataframe, i, prevstoploss):
        dataframe['Stoploss'] = np.nan
        if prevstoploss == 0:
            return 0.98 * dataframe['Close'][i]
        elif dataframe['Close'][i] < dataframe['Open'][i]:
            return prevstoploss
        elif dataframe['ADX'][i] < dataframe['ADX'][i - 1]:
            return 0.98 * dataframe['Open'][i]
        else:
            return 0.98 * dataframe['Open'][i]

    @staticmethod
    def calulatePotentialProfits(self, dataframe):
        buy = []
        sell = []
        profit = []
        stoploss = 0
        for i in range(dataframe.shape[0]):
            if len(buy) > len(sell):
                stoploss = self.setStopLoss(dataframe, i, stoploss)
            if dataframe['IsPositiveTrend'][i] == 1 and dataframe['MACDlarger'][i] == 1 and len(sell) == len(buy):
                buy.append(dataframe['Close'][i])
                print("Bought @:" + str(dataframe['Close'][i]))
                stoploss = self.setStopLoss(dataframe, i, stoploss)
                print("Stoplosvalue:" + str(stoploss))
            # elif dataframe['Stoploss'][i] > dataframe['Low'][i] and len(buy) > len(sell):
            elif stoploss > dataframe['Low'][i] and len(buy) > len(sell):
                sell.append(stoploss)
                print("Stopped loss @:" + str(stoploss))
                stoploss = 0
            elif dataframe['NegCrossover'][i] == 1 and len(buy) > len(sell):
                sell.append(dataframe['Close'][i])
                print("Sold @:" + str(dataframe['Close'][i]))
                stoploss = 0
        for i in range(len(sell)):
            profit.append(sell[i]-buy[i])
        print(buy)
        print(sell)
        print(profit)
        return

    @staticmethod
    def testProfits(dataframe):
        buy = []
        sell = []
        profit = []
        for i in range(dataframe.shape[0]):
            if dataframe['PosCrossover'][i] == 1 and len(sell) == len(buy):
                buy.append(dataframe['Close'][i])
                print("Bought @:" + dataframe['Close'][i])
            elif dataframe['Stoploss'][i] > dataframe['Low'][i] and len(buy) > len(sell):
                sell.append({dataframe['Stoploss'][i], dataframe['Open'][i]})
                print("Stopped loss @:" + dataframe['Stoploss'])
                print("Opened @:" + dataframe['Open'])
            elif dataframe['NegCrossover'][i] == 1 and len(buy) > len(sell):
                sell.append(dataframe['Close'][i])
                print("Sold @:" + dataframe['Close'][i])
        for i in range(len(sell)):
            profit.append(sell[i]-buy[i])
        print(buy)
        print(sell)
        print(profit)
        return
