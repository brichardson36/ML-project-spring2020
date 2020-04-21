import os
from scipy import misc
from matplotlib import pyplot as plt
from datetime import timedelta, date
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn import metrics
## Requires python 3.5-3.7, tensorflow, sklearn, etc.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def plot_data_scatter(data):
    data.plot(y='price', ax=axes[0,0])
    data.plot(y='vol', ax=axes[0,1])
    return None


def plot_data_kde(data):
    data.plot(y='price', ax=axes[1,0], kind = 'kde')
    data.plot(y='vol', ax=axes[1,1], kind = 'kde')
    return None


def plot_data(data,trading_days,time_open,time_close, resolution='month'):
    if resolution == 'day':
        for day in trading_days:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
            mask = (data.index>=day+time_open) & (data.index<=day+time_close)
            plot_data_scatter(data[mask])
            plot_data_kde(data[mask])
            plt.show()
    elif resolution == 'month':
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
        plot_data_scatter(data)
        plot_data_kde(data)
        plt.show()
    return None


def intraday_trade_bot_1(p1, p2, p1_pred, p2_pred):
    stock = 0
    profit = 0
    correct = 0
    incorrect = 0
    for i in range(len(p1)-5):
        if p2_pred[i] > (p1_pred[i]*1.0001):
            profit-= p1[i]
            stock+= 1
        elif p2_pred[i] < (p1_pred[i]) and stock > 0:
            profit+= p1[i]*stock
            stock = 0
        if (p2_pred[i] > (p1_pred[i]) and p2[i] > p1[i]) or (p2_pred[i] < p1_pred[i] and p2[i] < p1[i]):
            correct+=1 
        else:
            incorrect+=1
    profit+=np.mean(p2[-30:])*stock
    return stock, profit, correct, incorrect


def intraday_trade_bot_2(pred_price_diffs, prices, split_at):
    """
    Trade stocks based on predicted price differences.
    It 'trades' one stock per time interval.  

    Arguments:
    pred_price_diffs: array, or np.array, containing the predicted price diffs
    prices: array, or np.array, containing the true prices

    Returns: 
    stock_traded -- count of stocks traded
    profit -- total profit
    correct -- count of correct price movement predictions
    incorrect -- count of incorrect price movement predictions
    """
    stock_traded = 0
    profit = 0
    correct = 0
    incorrect = 0
    for i, pred_price_diff in enumerate(pred_price_diffs):
        if pred_price_diff > 0:
            profit+= price[i+split_at]-price[i-1+split_at]
            stock_traded+= 1
            if price[i+split_at] > price[i-1+split_at]:
                correct+= 1
            else:
                incorrect+=1 
    return stock_traded, profit, correct, incorrect


def predict(func, x_train, y1_train, y2_train):
    fit1 = func
    fit1.fit(x_train, y1_train)
    y1_pred = fit1.predict(x_test)
    fit2 = func
    fit2.fit(x_train, y2_train)
    y2_pred = fit2.predict(x_test)
    return y1_pred, y2_pred


def pd_to_nparray_time_as_epoch(pd_dfs):
    nparray = np.empty((pd_dfs[0]['price']['open'].size,1))
    for pd_df in pd_dfs:
        nparray_prices = pd_df['price'].values.astype(float)
        nparray_vol = pd_df['vol'].values.astype(float)
        new_nparray = np.concatenate([nparray_prices,nparray_vol], axis=1)
        nparray = np.concatenate((nparray,new_nparray), axis=1) 
    return nparray[:,1:]


def bin_data(data,day,time_open,time_close,bin_size='1min'):
    data_resampled = data.resample(bin_size).agg({'price': 'ohlc', 'vol': 'sum'})
    data_binned = pd.DataFrame(columns=data_resampled.columns)
    data_binned = data_resampled[(data_resampled.index>=day+time_open) & (data_resampled.index<=day+time_close)]
    return data_binned


def get_diffs(data, lambdas=5):
    data_diffs = []
    for l in range(lambdas):
        data_diffs.append(data.diff(periods=(l+1)).divide(data))
    return data_diffs


def normalize(x):
    x[np.isnan(x)] = 0.0
    x_centered = x - x.mean(axis=0)
    norm = np.linalg.norm(x_centered, axis=0)
    return x_centered/norm


def split_train_test(x,y,bin_size,train_test_ratio=1/5):
    num_of_obs = x.shape[0]
    indicies = np.arange(0,num_of_obs)
    split_at = int(num_of_obs*train_test_ratio-2)
    training_idx, test_idx = indicies[10 :split_at], indicies[split_at:-2]
    x_train, x_test = x[training_idx,:], x[test_idx,:]
    y1_train, y1_test = y[(training_idx+1)], y[(test_idx+1)]
    y2_train, y2_test = y[(training_idx+2)], y[(test_idx+2)]
    return x_train, x_test, y1_train, y1_test, y2_train, y2_test


def split_train_test(x,y,bin_size,train_test_ratio=1/5):
    num_of_obs = x.shape[0]
    indicies = np.arange(0,num_of_obs)
    split_at = int(num_of_obs*train_test_ratio-2)
    training_idx, test_idx = indicies[10 :split_at], indicies[split_at:-2]
    x_train, x_test = x[training_idx,:], x[test_idx,:]
    y1_train, y1_test = y[(training_idx+1)], y[(test_idx+1)]
    y2_train, y2_test = y[(training_idx+2)], y[(test_idx+2)]
    return x_train, x_test, y1_train, y1_test, y2_train, y2_test

def split_train_test_LSTM(data,bin_size,train_test_ratio=1/5):
    num_of_obs = data.shape[0]
    split_at = int(num_of_obs*train_test_ratio)
    x = []
    y = []
    i = 0
    while (i + bin_size) <= num_of_obs - 1:
        x.append(data[i:i+bin_size])
        y.append(data[i+bin_size])
        i+= 1
    x_train, y_train = np.array(x[:split_at]), np.array(y[:split_at])
    x_test, y_test = np.array(x[split_at:]), np.array(y[split_at:])
    return x_train, x_test, y_train, y_test


def LSTM_handler():
    return None


def LSTM_helper(input, output, state):
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
    memory_node = tf.tanh(tf.matmul(input, weights_memory_node) + tf.matmul(output, weights_memory_node_hidden) + bias_memory_node)
    state = state * forget_gate + input_gate * memory_node
    output = output_gate * tf.tanh(state)
    return state, output



## Select symbols
symbol_list = ['AAPL', 'ADBE', 'CRM', 'CSCO', 'INTC', 'MA', 'MSFT', 'NVDA', 'PYPL', 'V']
print('Starting stock analysis interface program...')
print('Available stocks:')
print(symbol_list)
print('On which stocks do you want to run your analysis?')
print('Example entry: AAPL MA MSFT')
symbols_to_analyze_input = input()
symbols_to_analyze = symbols_to_analyze_input.split()
print('Loading stocks {} into memory...'.format(symbols_to_analyze))
pickled_data = pd.DataFrame()
for symbol in symbols_to_analyze:
    if symbol_list.count(symbol) != 1:
        print('Error: ' + symbol + ' not found. Did you input the symbol correctly?')
    else:
        pickled_data = pickled_data.append(pd.read_pickle("./data/201605/{}.pkl".format(symbol)))
        print('Loaded ' + symbol)
price_by_symbol = dict(iter(pickled_data.groupby('symbol')))
print('Stocks loaded!')
print('Available stocks:')
symbol_list = symbols_to_analyze
print(symbol_list)

## Select time period
period_start = price_by_symbol[symbol_list[0]]['time'].min()
period_end = price_by_symbol[symbol_list[0]]['time'].max()
time_open = pd.Timedelta('09:30:00')
time_close = pd.Timedelta('16:00:00')
trading_days = pd.bdate_range(start=period_start, end=(period_end))
print('What bin size? (default = 1min)')
bin_size = '1min'

## Select regression techniques
print('Which methods do you want to run on the data? ')
#input('.....')
stocks_eod, profits, corrects, incorrects = [], [], [], []
## Perform regression techniques
methods = ['ridge','lstm']
for method in methods:
    if method == 'ridge':
        for symbol in symbol_list:
            price_by_symbol[symbol].index = price_by_symbol[symbol]['time']
            for day in trading_days:
                ohlc_vol_data_binned = bin_data(price_by_symbol[symbol],day,time_open,time_close,bin_size=bin_size)
                ohlc_vol_data_binned_diffs = get_diffs(ohlc_vol_data_binned)
                param = pd_to_nparray_time_as_epoch(ohlc_vol_data_binned_diffs)
                param = normalize(param)
                price = ohlc_vol_data_binned['price']['open'].values.astype(float)
                param = np.concatenate((price[:,np.newaxis],param), axis=1)
                x_train, x_test, y1_train, y1_test, y2_train, y2_test = split_train_test(param,price,bin_size,train_test_ratio=1/8)
                y1_pred, y2_pred = predict(Ridge(), x_train, y1_train, y2_train)
                stock_eod_day, profit_day, correct_day, incorrect_day = intraday_trade_bot_1(y1_test, y2_test, y1_pred, y2_pred)
                stocks_eod.append(stock_eod_day)
                profits.append(profit_day)
                corrects.append(correct_day)
                incorrects.append(incorrect_day)
                ## create numpy array lol...
    if method == 'lstm':
            price_by_symbol[symbol].index = price_by_symbol[symbol]['time']
            best_offsets = []
            for day in trading_days:
                num_of_bins_per_nn = 7
                ohlc_vol_data_binned = bin_data(price_by_symbol[symbol],day,time_open,time_close,bin_size=bin_size)
                ohlc_vol_data_binned_diffs = get_diffs(ohlc_vol_data_binned, 1)
                param = pd_to_nparray_time_as_epoch(ohlc_vol_data_binned_diffs)[:,1]
                param = normalize(param)
                param.shape
                x_train, x_test, y_train, y_test = split_train_test_LSTM(param,num_of_bins_per_nn,train_test_ratio=4/5)
                price = ohlc_vol_data_binned['price']['open']
                # Hyperparameters
                batch_size = 7
                window_size = 7
                hidden_layer = 256
                clip_margin = 4
                learning_rate = 0.001
                epochs = 30
                #place holders
                inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
                targets = tf.placeholder(tf.float32, [batch_size, 1])
                #weights for input gate
                weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
                weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
                bias_input = tf.Variable(tf.zeros([hidden_layer]))
                #weights for the forgot gate
                weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
                weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
                bias_forget = tf.Variable(tf.zeros([hidden_layer]))
                #weights for the output gate
                weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
                weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
                bias_output = tf.Variable(tf.zeros([hidden_layer]))
                #weights for the memory node
                weights_memory_node = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
                weights_memory_node_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
                bias_memory_node = tf.Variable(tf.zeros([hidden_layer]))
                #weights for the output layer
                weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
                bias_output_layer = tf.Variable(tf.zeros([1]))

                outputs = []
                for i in range(batch_size) : 
                    batch_state = np.zeros([1, hidden_layer], dtype=np.float32)
                    batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
                    for ii in range(window_size):
                        batch_state, batch_output = LSTM_helper(tf.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)
                    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
                losses = []

                for i in range(len(outputs)):
                    losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
                    
                loss = tf.reduce_mean(losses)
                gradients = tf.gradients(loss, tf.trainable_variables())
                clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
                session = tf.Session()
                session.run(tf.global_variables_initializer())
                for i in range(epochs):
                    traind_scores = []
                    ii = 0
                    epoch_loss = []
                    while(ii + batch_size) <= len(x_train):
                        x_batch = x_train[ii:ii+batch_size].reshape(num_of_bins_per_nn,num_of_bins_per_nn,1)
                        y_batch = y_train[ii:ii+batch_size].reshape(num_of_bins_per_nn,1)
                        o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs:x_batch, targets:y_batch})
                        epoch_loss.append(c)
                        traind_scores.append(o)
                        ii += batch_size
                    if (i % 30) == 0:
                        print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
                sup =[]
                for i in range(len(traind_scores)):
                    for j in range(len(traind_scores[i])):
                        sup.append(traind_scores[i][j][0])
                tests = []
                i = 0
                while i+batch_size <= len(x_test): 
                    o = session.run([outputs],feed_dict={inputs:x_test[i:i+batch_size].reshape(num_of_bins_per_nn,num_of_bins_per_nn,1)})
                    i += batch_size
                    tests.append(o)
                tests_new = []
                for i in range(len(tests)):
                  for j in range(len(tests[i][0])):
                    tests_new.append(tests[i][0][j])
                test_results = []
                length = param.shape[0]
                length_train = int(length*4/5)
                for i in range(length-9):
                      if i > (length_train):
                        test_results.append(tests_new[i-length_train])
                      else:
                        test_results.append(None)
                # plt.figure(figsize=(16,7))
                # plt.plot(param)
                # plt.plot(sup)
                # plt.plot(test_results)
                # plt.show()
                best_offset = 0
                best_profit = 0
                print("Offset: " + str(7))
                stock_traded_day, profit_day, correct_day, incorrect_day = intraday_trade_bot_2(tests_new,price, length_train+7 )
                print("Profit: " + str(profit_day))
                print("Correct: " + str(correct_day))
                print("Incorrect: " + str(incorrect_day))
            print('best offsets')
            print(best_offsets)
            print('avg best_offsets')
            print(np.median(best_offsets))
            print(np.mean(best_offsets))
print(stocks_eod)
print(profits)
print(corrects)
print(incorrects)
print(np.sum(stocks_eod), np.sum(profits), np.sum(corrects), np.sum(incorrects))


input('Press ENTER to exit')
