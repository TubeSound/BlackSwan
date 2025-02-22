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


        
def evaluate(symbol, timeframe, days=20):
    dirpath = f'./debug/squeezer/{symbol}/{timeframe}'
    os.makedirs(dirpath, exist_ok=True)
    data0 = from_pickle(symbol, timeframe)
    jst = data0[Columns.JST]
    t1 = jst[-1]
    t0 = t1 - timedelta(days=days)
    n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)   
    
    jst = data[Columns.JST]
    cl = data[Columns.CLOSE]
    
    SQUEEZER(data, 20, 2.0, 20)
    sqz = data[Indicators.SQUEEZER]
    std = data[Indicators.SQUEEZER_STD]
    atr = data[Indicators.SQUEEZER_ATR]

    fig, axes = makeFig(2, 1, (16, 10))
    axes[0].plot(jst, cl, color='blue', alpha=0.4)
    for i, s in enumerate(sqz):
        if s > 0:
            axes[0].scatter(jst[i], cl[i], color='red', alpha=0.2, s= 50)
            axes[0].hlines(cl[i], jst[0], jst[-1], color='red', alpha=0.5)
    axes[1].plot(jst, std, color='blue', alpha=0.4, label='std')
    axes[1].plot(jst, atr, color='red', alpha=0.4, label='atr')
    axes[1].legend()
    fig.savefig(os.path.join(dirpath,'imgge.png'))
    
 


    
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