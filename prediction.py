import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Indicators, Columns
from technical import PPP, detect_terms, is_nans
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader

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

def from_pickle(symbol, timeframe):
    import pickle
    if symbol == 'DOW' and timeframe == 'M15':
        filepath = './data/BacktestMarket/BM_dow_M15.pkl'
    elif symbol == 'NIKKEI' and timeframe == 'M15':
        filepath = './data/BacktestMarket/BM_nikkei_M15.pkl'
    else:
        filepath = './data/Axiory/' + symbol + '_' + timeframe + '.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

def timefilter(data, year_from, month_from, day_from, year_to, month_to, day_to):
    t0 = datetime(year_from, month_from, day_from).astimezone(JST)
    t1 = datetime(year_to, month_to, day_to).astimezone(JST)
    return TimeUtils.slice(data, data['jst'], t0, t1)

class MakeFeatures:
    
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        
    def make(self, technical_param:dict, pre:int, post:int, target:int):
        data = from_pickle(self.symbol, self.timeframe)
        p = technical_param['MA']
        PPP(self.timeframe, data, p['long_term'], p['mid_term'], p['short_term'])
        
        ma_long = data[Indicators.MA_LONG]
        ma_mid = data[Indicators.MA_MID]
        ma_short = data[Indicators.MA_SHORT]
        cl = data[Columns.CLOSE]
        gc = data[Indicators.MA_GOLDEN_CROSS]
        n = len(cl)
        up_terms = detect_terms(gc, 1)
        indices = []
        vectors = []
        prices = []
        for i0, i1 in up_terms:
            begin = i0 - pre
            end = i0 + target
            if i1 < end:
                continue
            if begin < 0 or end >= n:
                continue
            sl = slice(begin, i0 + post + 1)
            l = ma_long[sl]
            m = ma_mid[sl]
            s = ma_short[sl]
            if is_nans(l + m + s):
                continue
            normal = ma_mid[i0]
            l = (np.array(l) - normal) / normal * 100
            m = (np.array(m) - normal) / normal * 100
            s = (np.array(s) - normal) / normal * 100
            vectors.append([s, m, l])
            p = (cl[i0 + target] - cl[i0]) /cl[i0] * 100
            prices.append(p)
            indices.append(i0)
        return data, indices, vectors, prices
 
 
 

def plot(symbol, timeframe, values, pre, post, target):
     data, indices, vectors, prices = values
     cl = data[Columns.CLOSE]
     time = data[Columns.JST]
     ma_short = data[Indicators.MA_SHORT]
     ma_mid = data[Indicators.MA_MID]
     ma_long = data[Indicators.MA_LONG]
     dirpath = f'./debug/PPP/{symbol}/{timeframe}'
     os.makedirs(dirpath, exist_ok=True)
     page = 0
     for index, vector, price in zip(indices, vectors, prices):
        s, m, l = vector
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        begin = index - pre
        end = index + target
        sl = slice(begin, end + 1, 1)
        xlim = (begin, end)
        axes[0].plot(time[sl], cl[sl], color='purple')
        axes[0].scatter(time[sl], ma_short[sl], s=5, alpha=0.2, color='red')
        axes[0].scatter(time[sl], ma_mid[sl], s=5, alpha=0.2, color='green')
        axes[0].scatter(time[sl], ma_long[sl], s=5, alpha=0.2, color='blue')
        axes[0].scatter(time[index], cl[index], marker='o', color='green', s=100, alpha=0.5)
        axes[0].scatter(time[index + target], cl[index + target], marker='x', s=100, color='red', alpha=0.5)
        
        begin = index - pre
        sl = slice(begin, index + post + 1)
        axes[1].scatter(time[sl], s, s=5, alpha=0.5, color='red')
        axes[1].scatter(time[sl], m, s=5, alpha=0.5, color='green')
        axes[1].scatter(time[sl], l, s=5, alpha=0.5, color='blue')
        axes[1].hlines(0, time[xlim[0]], time[xlim[1]], color='gray')
        [ax.set_xlim(time[xlim[0]], time[xlim[1]]) for ax in axes]
        
        fig.savefig(os.path.join(dirpath, f'ma_graph_#{page}.png'))
        page += 1
        plt.close()
        
 
def main():
    symbol = 'DOW'
    timeframe = 'M5'
    making = MakeFeatures(symbol, timeframe)
    param = {'MA': {'long_term': 60, 'mid_term': 20, 'short_term': 5}}
    pre = 12 * 8
    post = 12 * 1
    target = 12 * 4
    values = making.make(param, pre, post, target)
    plot(symbol, timeframe, values, pre, post, target)
    
def test():
    a = [1, 2, 3, 4, 5]
    sl = slice(0, 3, 1)

    b = a[sl]
    print(a)
    print(b)
    
      
    
if __name__ == '__main__':
    main()