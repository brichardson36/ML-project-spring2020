import os
from scipy import misc
from matplotlib import pyplot as plt
from datetime import timedelta, date
import numpy as np
import pandas as pd
from datetime import datetime

data = pd.read_csv('171731d702aa6fcd.csv', parse_dates=[['DATE', 'TIME_M']])
data = data.drop(['SYM_SUFFIX'], axis=1)
data.columns = ['time', 'symbol', 'vol', 'price']
symbol_list = data['symbol'].unique()
data.groupby(['symbol']).plot(x='time', y='price')
#data.to_pickle("./data/20160501_sp500_top_ten_tech.pkl")
print(symbol_list)
price_by_symbol = dict(iter(data.groupby('symbol')))
for symbol in symbol_list:
	print(symbol)
	price_by_symbol[symbol].to_pickle("./data/201605/{}.pkl".format(symbol))