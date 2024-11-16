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

from common import Indicators, Columns, Signal
from technical import PPP, SUPERTREND, SUPERTREND_SIGNAL, MA, detect_terms, is_nans
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
 
 
 

def plot(symbol, timeframe, is_long,  data, values, pre, post, target):
     indices, vectors, prices = values
     cl = data[Columns.CLOSE]
     time = data[Columns.JST]
     ma_short = data[Indicators.MA_SHORT]
     ma_mid = data[Indicators.MA_MID]
     ma_long = data[Indicators.MA_LONG]
     cross =data[Indicators.MA_GOLDEN_CROSS]
     label = "long" if is_long else "short"
     dirpath = f'./debug/PPP/{symbol}/{timeframe}/{label}'
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
        axes[0].scatter(time[index], cl[index], marker='o', color='orange', s=100, alpha=0.5)
        axes[0].scatter(time[index + post], cl[index + post], marker='o', s=200, color='red', alpha=0.5)
        axes[0].scatter(time[index + target], cl[index + target], marker='o', s=300, color='red', alpha=0.5)
        
        if is_long:
            minv =  min(cl[sl])
            axes[0].set_ylim(minv, minv + 500)
        else:
            maxv = max(cl[sl])
            axes[0].set_ylim(maxv - 500, maxv)

        ax = axes[0].twinx()
        ax.plot(time[sl], cross[sl],alpha=0.5, color='orange')
        if is_long:
            ax.set_ylim(0, 5)
        else:
            ax.set_ylim(-5, 0)
        begin = index - pre
        sl = slice(begin, index + post + 1)
        axes[1].scatter(time[sl], s, s=5, alpha=0.5, color='red')
        axes[1].scatter(time[sl], m, s=5, alpha=0.5, color='green')
        axes[1].scatter(time[sl], l, s=5, alpha=0.5, color='blue')
        axes[1].hlines(0, time[xlim[0]], time[xlim[1]], color='gray')
        axes[1].scatter(time[index], m[pre], marker='o', color='orange', s=100, alpha=0.5)
        axes[1].scatter(time[index + post], m[pre + post], marker='o', s=200, color='red', alpha=0.5)
        axes[1].set_ylim(-0.5, 0.5)
        #axes[1].scatter(time[index + target], m[pre + target], marker='o', s=300, color='red', alpha=0.5)
        [ax.set_xlim(time[xlim[0]], time[xlim[1]]) for ax in axes]
        fig.savefig(os.path.join(dirpath, f'ma_graph_#{page}.png'))
        page += 1
        plt.close()
        
        
def plot2(strategy, symbol, timeframe, data0, df, dirpath, days=3):
    jst = data0[Columns.JST]
    tbegin = jst[0]
    tend = jst[-1]
    page = 0
    t = tbegin
    t1 = t + timedelta(days=days)
    while t < tend:
        n, data = TimeUtils.slice(data0, jst, t, t1)   
        time = data[Columns.JST]
        cl = data[Columns.CLOSE] 
    
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        axes[0].plot(time, cl, color='blue', alpha=0.2)
        if strategy == 'PPP':
            entries = data[Indicators.PPP_ENTRY]
            exits = data[Indicators.PPP_EXIT]
            ma_short = data[Indicators.MA_SHORT]
            ma_mid = data[Indicators.MA_MID]
            ma_long = data[Indicators.MA_LONG]
            axes[0].scatter(time, ma_short, alpha=0.2, color='red', marker='o', s= 5)
            axes[0].scatter(time, ma_mid, alpha=0.2, color='green', marker='o', s= 5)
            axes[0].scatter(time, ma_long, alpha=0.2, color='blue', marker='o', s= 5)
        elif strategy == 'supertrend':
            entries = data[Indicators.SUPERTREND_ENTRY]
            exits = data[Indicators.SUPERTREND_EXIT]
            ma = data[Indicators.SUPERTREND_MA]
            up = data[Indicators.SUPERTREND_U]
            down = data[Indicators.SUPERTREND_L]
            axes[0].scatter(time, up, alpha=0.6, color='green', marker='o', s= 5)
            axes[0].scatter(time, down, alpha=0.4, color='orange', marker='o', s= 5)
            axes[0].scatter(time, ma, alpha=0.4, color='red', marker='o', s= 5)
            
        for i, entry in enumerate(entries):
            if entry == 1:
                color = 'green'
            elif entry == -1:
                color= 'red'
            else:
                continue
            axes[0].vlines(time[i], min(cl), max(cl), lw=2, linestyle='dotted', alpha=0.4, color=color)
            
        entry_count = 0
        exit_count = 0        
        for i in range(len(df)):
            record = df.iloc[i, :]
            sig = record['signal']
            if sig == Signal.LONG:
                color = 'green'
                marker1 = '^'
            elif sig == Signal.SHORT:
                color = 'red'
                marker1 = 'v'
            profit = record['profit']
            t_entry_str = record['entry_time']
            t_entry = datetime.strptime(t_entry_str[:16], '%Y-%m-%d %H:%M').astimezone(JST)
            ylim = axex[0].get_ylim()
            if t_entry >= time[0] and t_entry <= time[-1]:
                entry_count += 1
                axes[0].scatter(t_entry, record['entry_price'], color=color, alpha=0.4, marker=marker1, s=200)
                dy = (entry_count % 3) * ylim / 20 
                axes[0].text(t_entry, min(cl) + dy, f'e{i}')        
            t_exit_str = record['exit_time']
            t_exit = datetime.strptime(t_exit_str[:16], '%Y-%m-%d %H:%M').astimezone(JST)
            if profit > 0:
                p = f'+{profit}'
                marker2 = 'o'
            else:
                p = f'{profit}'
                marker2 = 'x'
            if t_exit >= time[0] and t_exit <= time[-1]:
                axes[0].scatter(t_exit, record['exit_price'], color=color, alpha=0.4, marker=marker2, s=300)
                exit_count += 1
                dy = (exit_count % 3) * ylim / 20 
                axes[0].text(t_exit, max(cl) - dy, f'c{i}{p}')

        
        if strategy == 'PPP':
            up = data[Indicators.PPP_UP]
            down = data[Indicators.PPP_DOWN]
            ax = axes[0].twinx()
            ax.set_ylim(-2, 2)
            axes[1].plot(time, -1 * np.array(down), alpha=0.5, color='red')
            axes[1].plot(time, up, alpha=0.5, color='green')
            axes[1].set_ylim(-2, 2)
            #axes[1].scatter(time[index + target], m[pre + target], marker='o', s=300, color='red', alpha=0.5)
            [ax.set_xlim(time[0], time[-1]) for ax in axes]
        fig.savefig(os.path.join(dirpath, f'ma_graph_#{page}.png'))
        page += 1
        plt.close()        
        t = t1
        t1 = t + timedelta(days=days)
        
        
def trade_ppp(symbol, timeframe, data, technical_param, trade_param):
    p1 = technical_param['MA']
    p2 = technical_param['PPP']
    PPP(timeframe, data, p1['long_term'], p1['mid_term'], p1['short_term'], p2['pre'], p2['post'], p2['target'])
    sim = Simulation(data, trade_param)        
    df, summary, profit_curve = sim.run(data, Indicators.PPP_ENTRY, Indicators.PPP_EXIT)
    trade_num, profit, win_rate = summary
    return (df, summary, profit_curve)

def trade_supertrend(symbol, timeframe, data, technical_param, trade_param):
    p = technical_param
    SUPERTREND(data, p['atr_window'], p['atr_multiply'])
    SUPERTREND_SIGNAL(data, p['short_term'])
    sim = Simulation(data, trade_param)        
    df, summary, profit_curve = sim.run(data, Indicators.SUPERTREND_ENTRY, Indicators.SUPERTREND_EXIT)
    trade_num, profit, win_rate = summary
    return (df, summary, profit_curve)


def make_technical_param_ppp(randomize=False):
    def gen3():
        s = m = l = 0
        while s >= m or m >= l:
            s = random.randint(1, 5) * 5
            m = random.randint(2, 10) * 5
            l = random.randint(3, 20) * 5
        return s, m, l    
    
    if randomize:
        short_term, mid_term, long_term = gen3()
        post = random.randint(1, 6) * 6
    else:
        short_term = 5
        mid_term = 20
        long_term = 60
        post = 12 * 1
    
    param = {'MA': {'long_term': long_term, 'mid_term': mid_term, 'short_term': short_term}}
    pre = 12 * 4
    target = 12 * 16
    param['PPP'] = {'pre': pre, 'post': post, 'target': target}
    return param

def make_technical_param_ppp(randomize=False):
    def gen3():
        s = m = l = 0
        while s >= m or m >= l:
            s = random.randint(1, 5) * 5
            m = random.randint(2, 10) * 5
            l = random.randint(3, 20) * 5
        return s, m, l    
    
    if randomize:
        short_term, mid_term, long_term = gen3()
        post = random.randint(1, 6) * 6
    else:
        short_term = 5
        mid_term = 20
        long_term = 60
        post = 12 * 1
    
    param = {'MA': {'long_term': long_term, 'mid_term': mid_term, 'short_term': short_term}}
    pre = 12 * 4
    target = 12 * 16
    param['PPP'] = {'pre': pre, 'post': post, 'target': target}
    return param


def make_technical_param_supertrend(randomize=False):
    if randomize:
        window = random.randint(10, 100)
        multiply = random.random() * 3 + 0.5
        ma = random.randint(10, 100)
        term = random.randint(4, 50)
    else:
        window = 40
        multiply = 1.4
        ma = 40
        term = 12
    param = {  
                'atr_window': window, 
                'atr_multiply': multiply,
                'ma_window': ma,
                'short_term': term,
            }
    return param
    
def make_trade_param(symbol, randomize=False):
    if symbol == 'XAUUSD':
        k = 0.1
    elif symbol == 'NSDQ':
        k = 0.5
    else:
        k = 1.0
    
    if randomize:
        sl = random.randint(1, 10) * 50
        target_profit = random.randint(1, 10) * 50
        trail_stop = random.randint(1, 5) * 50
    else:
        sl = 250
        target_profit = 300
        trail_stop = 200
    
    param =  {
                'strategy': 'supertrend',
                'begin_hour': 0,
                'begin_minute': 0,
                'hours': 0,
                'sl': {
                        'method': Simulation.SL_FIX,
                        'value': int(sl * k)
                    },
                'target_profit': int(target_profit * k),
                'trail_stop': int(trail_stop * k), 
                'volume': 0.1, 
                'position_max': 10, 
                'timelimit': 0}
    return param, k
 
def test(symbol, timefram, strategy):
    #making = MakeFeatures(symbol, timeframe)
    dirpath = f'./test/{strategy}/{symbol}/{timeframe}'
    os.makedirs(dirpath, exist_ok=True)

    data0 = from_pickle(symbol, timeframe)
    jst = data0[Columns.JST]
    t1 = jst[-1]
    t0 = t1 - timedelta(days=30*6)
    n, data = TimeUtils.slice(data0, jst, t0, t1)   
    
    trade_param, k = make_trade_param(symbol)
    if strategy.find('supertrend') >= 0:
        technical_param = make_technical_param_supertrend()
        result = trade_supertrend(symbol, timeframe, data, technical_param, trade_param)
    (df, summary, profit_curve) = result
    print(summary)
    df.to_csv(os.path.join(dirpath, 'trade_summary.csv'), index=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(profit_curve[0], profit_curve[1])
    fig.savefig(os.path.join(dirpath, 'profit.png'))
    plot2(strategy, symbol, timeframe, data, df, dirpath)
    
def optimize(symbol, timefram, strategy):
    #making = MakeFeatures(symbol, timeframe)
    dirpath = f'./optimize_all/{strategy}/{symbol}/{timeframe}'
    os.makedirs(dirpath, exist_ok=True)

    data = from_pickle(symbol, timeframe)
    #jst = data0[Columns.JST]
    #t0 = jst[0]
    #t1 = t0 + timedelta(days=180)
    #n, data = TimeUtils.slice(data0, jst, t0, t1)   
    jst = data[Columns.JST]
    print('Data length', len(jst), jst[0], jst[-1])

    out = []
    for i in range(1000):
        trade_param, k = make_trade_param(symbol, randomize=True)
        if strategy.find('supertrend') >= 0:
            technical_param = make_technical_param_supertrend(randomize=True)
            result = trade_supertrend(symbol, timeframe, data, technical_param, trade_param)
        (df, summary, profit_curve) = result
        trade_num, profit, win_rate = summary
        d1, columns1 = expand('p1', technical_param)
        d2, columns2 = expand('p2', trade_param)
        d = [i] + d1 + d2 + summary 
        out.append(d)
        
        print(i, summary)
        if profit > 15000 * k:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(profit_curve[0], profit_curve[1])
            fig.savefig(os.path.join(dirpath, f'{symbol}_{timeframe}_profit#{i}.png'))
     
        try:
            columns = ['no'] + columns1 + columns2 + ['trade_num', 'profit', 'win_rate']
            df = pd.DataFrame(data=out, columns=columns)
            df.to_csv(os.path.join(dirpath, 'trade_summary.csv'), index=False)
        except:
            continue
    
if __name__ == '__main__':
    args = sys.argv
    if len(args) != 4:
        symbol = 'XAUUSD'
        timeframe = 'M15'
        strategy = 'supertrend'
    else:        
        symbol = args[1]
        timeframe = args[2]
        if args[3] == 'su':
            strategy = 'supertrend'
        elif args[3] == 'ppp':
            strategy = 'PPP'
        
    prinnt(symbol, timeframe, strategy)
    #test(symbol, timeframe, strategy)
    optimize(symbol, timeframe, strategy)