import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Indicators, Columns, Signal
from technical import SQUEEZER
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader
import random

def makeFig(rows, cols, size):
    fig, ax = plt.subplots(rows, cols, figsize=(size[0], size[1]))
    return (fig, ax)

def gridFig(row_rate, size):
    rows = sum(row_rate)
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(rows, 1, hspace=0.6)
    axes = []
    begin = 0
    for rate in row_rate:
        end = begin + rate
        ax = plt.subplot(gs[begin: end, 0])
        axes.append(ax)
        begin = end
    return (fig, axes)

def expand(name: str, dic: dict):
    data = []
    columns = []
    for key, value in dic.items():
        if name == '':
            column = key
        else:
            column = name + '.' + key
        if type(value) == dict:
            d, c = expand(column, value)                    
            data += d
            columns += c
        else:
            data.append(value)
            columns.append(column)
    return data, columns 

def from_pickle(symbol, timeframe):
    import pickle
    #if symbol == 'DOW' and timeframe == 'M15':
    #    filepath = './data/BacktestMarket/BM_dow_M15.pkl'
    #elif symbol == 'NIKKEI' and timeframe == 'M15':
    #    filepath = './data/BacktestMarket/BM_nikkei_M15.pkl'
    #else:
    filepath = './data/Axiory/' + symbol + '_' + timeframe + '.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

def timefilter(data, year_from, month_from, day_from, year_to, month_to, day_to):
    t0 = datetime(year_from, month_from, day_from).astimezone(JST)
    t1 = datetime(year_to, month_to, day_to).astimezone(JST)
    return TimeUtils.slice(data, data['jst'], t0, t1)


def plot_marker(ax, data, signal, markers, colors, alpha=0.5, s=50):
    time = data[Columns.JST]
    cl = data[Columns.CLOSE]
    for i, status in enumerate(signal):
        if status == 1:
            color = colors[0]
            marker = markers[0]
        elif status == -1:
            color = colors[1]
            marker = markers[1]
        else:
            continue
        ax.scatter(time[i], cl[i], color=color, marker=marker, alpha=alpha, s=s)
        
def evaluate(symbol, timeframe, days=10):
    dirpath = f'./debug/squeezer/{symbol}/{timeframe}'
    os.makedirs(dirpath, exist_ok=True)
    data0 = from_pickle(symbol, timeframe)
    time = data0[Columns.JST]
    t0 = time[0]
    t1 = t0 + timedelta(days=days)

    SQUEEZER(data0, 20, 2.0, 100)
    count = 0
    while t1 < time[-1]:
        n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)   
        jst = data[Columns.JST]
        cl = data[Columns.CLOSE]
    
        sqz = data[Indicators.SQUEEZER]
        std = data[Indicators.SQUEEZER_STD]
        atr = data[Indicators.SQUEEZER_ATR]
        upper = data[Indicators.SQUEEZER_UPPER]
        lower = data[Indicators.SQUEEZER_LOWER]
        signal = data[Indicators.SQUEEZER_SIGNAL]
        entry = data[Indicators.SQUEEZER_ENTRY]
        ext = data[Indicators.SQUEEZER_EXIT]

        fig, axes = makeFig(2, 1, (16, 10))
        axes[0].plot(jst, cl, color='blue', alpha=0.4)
        plot_marker(axes[0], data, signal, ['o', 'o'], ['green', 'red'], s=150)
        axes[0].plot(jst, upper, color='green', alpha=0.5)       
        axes[0].plot(jst, lower, color='orange', alpha=0.5) 
        plot_marker(axes[0], data, entry, ['^', 'v'], ['green', 'red'], s=100)
        plot_marker(axes[0], data, ext, ['x'], ['gray'], s=200, alpha=0.2)
        axes[1].plot(jst, std, color='blue', alpha=0.4, label='std')
        axes[1].plot(jst, atr, color='red', alpha=0.4, label='atr')
        axes[1].legend()
        fig.savefig(os.path.join(dirpath, f'#{count}_imgge.png'))
        plt.close()
        count += 1
        t0 = t1
        t1 = t0 + timedelta(days=days)


    
if __name__ == '__main__':
    args = sys.argv
    if len(args) != 4:
        symbol = 'NSDQ'
        timeframe = 'M5'
        strategy = 'supertrend'
    else:        
        symbol = args[1]
        timeframe = args[2]
        if args[3] == 'su':
            strategy = 'supertrend'
        elif args[3] == 'ppp':
            strategy = 'PPP'
        
    print(symbol, timeframe, strategy)
    evaluate(symbol, timeframe)