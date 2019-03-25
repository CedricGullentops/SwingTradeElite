import pandas as pd
import quandl
quandl.ApiConfig.api_key = "H1T2ddaL3XRzU4afn4vZ"
import datetime
import matplotlib.pyplot as plt

# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2016, 1, 1)
print start
start = (datetime.datetime.today() + datetime.timedelta(-6*365/12)).strftime("%Y-%m-%d %H:%M:%S")
print start
end = datetime.date.today()
print end

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
s = "TSLA"
apple = quandl.get("WIKI/" + s, start_date=start, end_date=end)

print apple

print apple.head()

apple["Adj. Close"].plot(grid=True)

