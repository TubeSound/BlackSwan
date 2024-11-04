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
from technical import PPP
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
        
class BackTest:
    def __init__(self, name, symbol, timeframe, strategy, data):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy.upper()
        self.data = data
        
  
    def trade(self, trade_param, technical_param):
        data = self.data.copy()
        self.indicators(data, technical_param)
        sim = Simulation(trade_param)        
        df, summary, profit_curve = sim.run(data, 'MABAND_LONG', 'MABAND_SHORT')
        trade_num, profit, win_rate = summary
        return (df, summary, profit_curve)

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
  
class Optimizer:
    def __init__(self, data, number : int, symbol: str, timeframe: float, repeat: int):
        self.data = data
        self.number = number
        self.symbol = symbol
        self.timeframe = timeframe
        

        
        
    def get_dirpath(self):
        return './2018.10-2024.5/'
        
        
    def get_path(self, number: int, year=None, month=None):
        dirpath = os.path.join(self.get_dirpath(), 'result')
        os.makedirs(dirpath, exist_ok=True)
        if year is None:
            filename = str(number).zfill(2) + '_' +  self.strategy + '_' + self.symbol + '_' + self.timeframe + '.xlsx'
        else:
            filename = str(number).zfill(2) + '_' +  self.strategy + '_' + self.symbol + '_' + self.timeframe + '_' + str(year) + '_' + str(month).zfill(2) + '.xlsx'
        path = os.path.join(dirpath, filename)      
        return path
          

        
    
    def run(self):
        year_from = 2018
        month_from = 10
        year_to = 2024
        month_to = 7
        n, data = self.from_pickle(year_from, month_from, year_to, month_to)
        if n < 100:
            print('Data size small')
            return
        self.trade(data)
        
    def trade(self, data, year=None, month=None):
        symbol = self.symbol
        param = Parameters(symbol, self.strategy)
        result = []
        for i in range(self.repeat):
            trade_param, technical_param = param.generate()                
            test = BackTest('', symbol, self.timeframe, self.strategy, data)
            df, summary, profit_curve = test.trade(trade_param, technical_param)
            d, columns= self.arrange_data(i, symbol, self.timeframe, technical_param, trade_param, summary)
            result.append(d)
            _, profit, _ = summary
            print(i, 'Profit', summary)
            if profit > 10000:
                self.save_profit_curve(symbol, self.timeframe, i, profit_curve, year, month)                    
            try:
                df = pd.DataFrame(data=result, columns=columns)
                df = df.sort_values('profit', ascending=False)
                path = self.get_path(self.number, year=year, month=month)
                df.to_excel(path, index=False)
            except:
                pass
            
    def arrange_data(self, i, symbol, timeframe, technical_param, trade_param, summary):
        data = []
        columns = ['No', 'symbol', 'timeframe']
        data += [i, symbol, timeframe]
        for param in [technical_param, trade_param]:
            d, c = expand('', param)
            data += d
            columns += c
        columns += ['trade_num', 'profit', 'win_rate']    
        data += list(summary)
        return data, columns
    
    def save_profit_curve(self, symbol, timeframe, i, profit_curve, year, month):
        dirpath = os.path.join(self.get_dirpath(), 'profit_curve', self.strategy,  symbol, timeframe, str(self.number).zfill(2))
        os.makedirs(dirpath, exist_ok=True)
        if year is None:
            filename = '#' + str(i).zfill(4) + '_' + self.strategy + '_' + symbol + '_' + timeframe + '.png'
        else:
            filename = '#' + str(i).zfill(4) + '_' + self.strategy + '_' + symbol + '_' + timeframe + str(year) + '_' + str(month).zfill(2) + '.png'
        path = os.path.join(dirpath, filename)
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        if year is None:
            title = '#' + str(i).zfill(4) +  ' ' + self.strategy + ' ' + symbol + ' ' + timeframe
        else:
            title = '#' + str(i).zfill(4) +  ' ' + self.strategy + ' ' + symbol + ' ' + timeframe + '(' + str(year) + '-' + str(month).zfill(2) + ')'
            
        ax.plot(range(len(profit_curve)), profit_curve, color='blue')
        ax.set_title(title)
        try:
            plt.savefig(path)   
            plt.close()               
        except:
            pass      
def opt():
    args = sys.argv
    if len(args) == 5:
        symbol = args[1]
        timeframe = args[2]
        strategy = args[3]
        number = args[4]
    elif len(args) == 1:
        symbol = 'DOW'
        timeframe = 'M15'
        strategy = 'SUPERTREND'
        number = 0
    else:
        raise Exception('Bad parameter')
    repeat = 1000
    opt = Optimizer(number, symbol, timeframe, strategy, repeat)
    opt.run()
    
    
def get_trade_param(sl, trailing_target, trailing_stop):
    param =  {
                'strategy': 'supertrend',
                'begin_hour': 0,
                'begin_minute': 0,
                'hours': 0,
                'sl': {
                        'method': 'fix',
                        'value': sl
                    },
                'target_profit': trailing_target,
                'trailing_stop': trailing_stop, 
                'volume': 0.1, 
                'position_max': 5, 
                'timelimit': 0}
    return param


def plot(data):
    jst = data['jst']
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']


    ma_short = data['MA_SHORT']
    limit_upper = data[Indicators.SUPERTREND_UPPER]
    limit_lower = data[Indicators.SUPERTREND_LOWER]
    atr_u = data[Indicators.ATR_UPPER]
    atr_l = data[Indicators.ATR_LOWER]
    signal = data[Indicators.SUPERTREND_SIGNAL]
    #profit, df, curve = calc_profit(data, long_event, short_event)
    #print(df)
    #drawdown = calc_drawdown(curve[1])
    #print(df)
    #print('Profit:', profit, drawdown)
    
    #axes[0].plot(jst, cl, color='blue')
    fig, axes = gridFig([4, 1], (15, 8))
    axes[0].plot(jst, ma_short, color='blue')
    axes[0].plot(jst, atr_u, color='gray', linewidth=1.0)
    axes[0].plot(jst, atr_l, color='gray', linewidth=1.0)
    axes[0].plot(jst, limit_upper, color='orange', linewidth=2.0)
    axes[0].plot(jst, limit_lower, color='green', linewidth=2.0)
    
    
    for i, s in enumerate(signal):
        if s == 1:
            axes[0].scatter(jst[i], ma_short[i], color='green', marker='o', s=200, alpha=0.5)
        if s == -1:
            axes[0].scatter(jst[i], ma_short[i], color='red', marker='o', s=200, alpha=0.8)

  
    #axes[2].plot(curve[0], curve[1])
    
    [ax.set_xlim(jst[0], jst[-1]) for ax in axes]    
    form = mdates.DateFormatter("%m/%d %H:%M")
    [ax.xaxis.set_major_formatter(form) for ax in axes]
    
    
def plot_profit(path, number, param, curve):
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(curve[0], curve[1], color='blue')
    ax.set_title("#" + str(number) + ' ' + str(param))
    fig.savefig(path)
    plt.close()
    
def trade(symbol, timeframe, param, data, sl, trailing_target, trailing_stop):
    trade_param = get_trade_param(sl, trailing_target, trailing_stop)
    sim = Simulation(data, trade_param)        
    return sim.run_doten(Indicators.PPP_ENTRY, Indicators.PPP_EXIT)
    #return sim.run_trailing_stop(Indicators.MAGAP_ENTRY)

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
    
        
def fulltime(symbol, timeframe):
    print(symbol, timeframe)
    data0 = from_pickle(symbol, timeframe)
    root = f'./ppp'
    data = data0.copy()
    title = f'{symbol}_{timeframe}_ppp'
    sim(root, symbol, timeframe, title, data)    

def sim(root, symbol, timeframe, title, data):
    dir_path = os.path.join(root, f'{symbol}/{timeframe}')
    os.makedirs(dir_path, exist_ok=True)
    out = []
    long_term = 288
    short_term = 4
    tap = 8
    threshold = 0.1    

    param = {'long_term': long_term, 'mid_term': mid_term, 'short_term': short_term, 'slope_tap': tap, 'slope_threshold': threshold}
    PPP(timeframe, data, long_term, mid_term, short_term, tap)
    MAGAP_SIGNAL(timeframe, data, threshold)
    r = trade(symbol, timeframe, param, data)
    if r is not None:
        jst = data['jst']
        df, summary, curve = r
        p, columns = expand('magap', param)
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        ax[0].plot(jst, data['close'])
        ax[1].plot(curve[0], curve[1])
        path = f'profit_curve_{title}.png'
        fig.savefig(os.path.join(dir_path, path))
        
        
def optimize(symbol, timeframe):
    print(symbol, timeframe)
    data0 = from_pickle(symbol, timeframe)
    root = f'./ppp_2024-7'
    year = 2024
    month_from = 7
    month_to = 9
    n, data = timefilter(data0, year, month_from, 1, year, month_to, 30)
    limit = 1500
    title = f'{symbol}_{timeframe}_magap'    
    sim_opt(root, symbol, timeframe, title, data, limit)    


def sim_opt(root, symbol, timeframe, title, data, limit):
    dir_path = os.path.join(root, f'{symbol}/{timeframe}')
    os.makedirs(dir_path, exist_ok=True)
    
    coeff = {'M5': 1, 'M15': 2, 'M30': 4}
    k = coeff[timeframe]
    
    stop_loss = 100
    trailing_target = 200
    trailing_stop = 100
    
    number = 0
    out = []
    for p1 in [1, 2, 3, 4, 5, 6, 7, 8]:
        long_term = int(p1 * 2 * 24 * k)
        for p2 in [1, 2, 3, 4]:
            if p2 > p1:
                continue
            mid_term = int(p2 * 2 * 24 * k)
            for p3 in range(2, 40, 2):
                short_term = int(p3 * 2 * k)
                if short_term >= mid_term:
                    continue       
                for tap in range(0, 10):
                    number += 1
                    param = {'long_term': long_term,
                             'mid_term': mid_term,
                             'short_term': short_term,
                             'tap': tap}
                    PPP(timeframe, data, param['long_term'], param['mid_term'], param['short_term'], tap = param['tap'])
                    r = trade(symbol, timeframe, param, data, stop_loss, trailing_target, trailing_stop)
                    if r is None:
                        continue
                    df , summary, curve = r
                    p, columns = expand('magap', param)
                    out.append([number] + p + list(summary))
                    if summary[0] > 10 and summary[1]> limit:
                        print(summary)
                        print(param)
                        path = f'profit_curve_#{number}_{title}.png'
                        plot_profit(os.path.join(dir_path, path), number, param, curve)
    df = pd.DataFrame(data=out, columns=['no'] + columns + ['n', 'profit', 'drawdown'])  
    df = df.sort_values('profit', ascending=False)         
    path = os.path.join(dir_path, f'summary_{title}.xlsx')
    df.to_excel(path, index=False)           


    
def vis(num, symbol, timeframe, data, technical_param, trade_param):
    #print(symbol, timeframe)
    title = f'{symbol}_{timeframe}_ppp'    
    
    
    jst = data['jst']
    cl = data[Columns.CLOSE]
    ma = data[Indicators.MA_MID]
    fig, ax = plt.subplots(4, 1, figsize=(15, 15))
    ax[0].plot(jst, cl)
    ax[1].plot(jst, data[Indicators.MA_LONG], color='purple')
    ax[1].plot(jst, data[Indicators.MA_MID], color='blue')
    ax[1].plot(jst, data[Indicators.MA_SHORT], color='red')
    slope = data[Indicators.MA_MID_SLOPE]
    ax[2].plot(jst, data[Indicators.ATRP], color='orange')
    ax[2].hlines(0, jst[0], jst[-1], color='gray')
    ax[2].set_ylim(0, 0.5)
    ax2 = ax[2].twinx()
    ax2.plot(jst, data[Indicators.ATR], color='red')
    ax[3].plot(jst, slope, color='green')
    ax[3].hlines(0, jst[0], jst[-1], color='gray')
    ax[3].set_ylim(-0.1, 0.1)
    #ax[2].hlines(0, jst[0], jst[-1], color='yellow')
    
    entry = data[Indicators.PPP_ENTRY]
    for i, v in enumerate(entry):
        if v == 1:
            ax[0].scatter(jst[i], cl[i], marker='^', color='green', s=200)
            ax[1].scatter(jst[i], ma[i], marker='^', color='green', s=200)
            
        elif v == -1:
            ax[0].scatter(jst[i], cl[i], marker='v', color='red', s=200)
            ax[1].scatter(jst[i], ma[i], marker='v', color='red', s=200)
    ext = data[Indicators.PPP_EXIT]
    for i, v in enumerate(ext):
        if v == 1:
            ax[0].scatter(jst[i], cl[i], marker='x', color='gray', s=300)
            ax[1].scatter(jst[i], ma[i], marker='x', color='gray', s=300)

    
    root = './PPP'
    os.makedirs(root, exist_ok=True)
    path = f'{symbol}_{timeframe}_#{num}_entry_exit.png'
    fig.savefig(os.path.join(root, path))
    plt.close()
    





def optimize_crash(symbol, timeframe):
    print(symbol, timeframe)  
    data0 = from_pickle(symbol, timeframe)
    for ma_window in range(10, 60, 5):
        root = f'./optimize/Axiory/covid/MA{ma_window}'
   
        year = 2020
        month_from = 1
        month_to = 4
        n, data = timefilter(data0, year, month_from, year, month_to)
        title = f'{symbol}_{timeframe}_MA{ma_window}_{year}_{month_from}'
        sim(root, symbol, timeframe, title, data, ma_window, 5000)
        
        
        
def technical_param(symbol):
    param_nikkei = {'PPP': {
                        'long_term': 240,
                        'mid_term': 144,
                        'short_term': 55,
                        'tap': 0,
                        'threshold': 0.01
                    }
            }
    if symbol.lower() == 'nikkei':
        return param_nikkei
    
    param_dow = {'PPP': {
                        'long_term': 384,
                        'mid_term': 192,
                        'short_term': 68,
                        'tap': 0,
                        'threshold': 0.01
                    }
            }
    if symbol.lower() == 'dow':
        return param_dow

def trade_param():
   param = {'begin_hour':8, 
                  'begin_minute':0,
                  'hours': 24,
                  'sl': {'method': 1, 'value':100},
                  'volume': 0.1,
                  'position_max':5,
                  'trail_target':100, 
                  'trail_stop': 50,
                  'timelimit':0}
   return param


def main1():
    args = sys.argv
    if len(args) == 3:
        symbol = args[1]
        timeframe = args[2]
    else:
        symbol = 'NIKKEI'
        timeframe = 'M5'
            
    data0 = from_pickle(symbol, timeframe)
    param = technical_param(symbol)['PPP']
    PPP(timeframe, 
        data0, 
        param['long_term'], 
        param['mid_term'], 
        param['short_term'],
        tap=param['tap'],
        threshold=param['threshold'])
    
    jst = data0[Columns.JST]
    tbegin = datetime(2024, 10, 10).astimezone(JST) #jst[0]
    tend = datetime(2024, 10, 12).astimezone(JST) #jst[-1]
    n, data = TimeUtils.slice(data0, jst, tbegin, tend)    
    r = trade(symbol, timeframe, param, data, 100, 100, 50)
    if r is not None:
        df , summary, curve = r
        df.to_csv(f'./PPP/trade_{symbol}_{timeframe}.csv', index=False)
    print(symbol, timeframe, n)
    vis(0, symbol, timeframe, data, param, trade_param())
    
    
def main2():
    args = sys.argv
    if len(args) == 3:
        symbol = args[1]
        timeframe = args[2]
    else:
        symbol = 'NIKKEI'
        timeframe = 'M5'
    optimize(symbol, timeframe)
    #fulltime(symbol, timeframe)
    
def main3():
    args = sys.argv
    if len(args) == 3:
        symbol = args[1]
        timeframe = args[2]
    else:
        symbol = 'NIKKEI'
        timeframe = 'M5'
            
    data0 = from_pickle(symbol, timeframe)
    param = technical_param(symbol)['PPP']
    PPP(timeframe, 
        data0, 
        param['long_term'], 
        param['mid_term'], 
        param['short_term'],
        tap=param['tap'],
        threshold=param['threshold'])
    
    jst = data0[Columns.JST]
    tbegin = jst[0]
    tend = jst[-1]
    t = tbegin
    i = 0
    while t < tend:
        t1 = t + timedelta(days=14)
        n, data = TimeUtils.slice(data0, jst, t, t1)    
        print(symbol, timeframe, n)
        vis(i, symbol, timeframe, data, param, trade_param())
        t = t1
        i += 1
    
if __name__ == '__main__':
    main3()
    #test()