import os
import shutil
import sys
import pickle
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
from technical import PPP, SUPERTREND, SUPERTREND_SIGNAL, MA, detect_terms, is_nans, sma, breakout, rally
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

def from_pickle(symbol):
    filepath = f'./data/Axiory/tick/{symbol}_TICK_2024_12.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
        jst = list(data0['jst'])
        bid = list(data0['bid'])
        ask = list(data0['ask'])
        timestamp = [t.timestamp() for t in jst]
        dic = {Columns.JST: jst, Columns.TIMESTAMP: timestamp, Columns.BID: bid, Columns.ASK: ask}
        print(symbol, 'data from ', jst[0], 'to',  jst[-1])
        return dic        
    return None

def timefilter(data, year_from, month_from, day_from, year_to, month_to, day_to):
    t0 = datetime(year_from, month_from, day_from).astimezone(JST)
    t1 = datetime(year_to, month_to, day_to).astimezone(JST)
    jst = data[Columns.JST]
    return TimeUtils.slice(data, jst, t0, t1)

def search(time, index, tref):
    n = len(time)
    for i in range(index, n):
        if time[i] >= tref:
            return i
    return -1

def search_reverse(time, index, tref):
    n = len(time)
    for i in range(index, -1, -1):
        if time[i] <= tref:
            return i
    return -1

def search_head(time, minutes):
    i = search(time, 0, time[0] + 60 * minutes)
    return i


def range_minutes(time, array, minutes):
    n = len(time)
    rng = np.full(n, 0)
    counts = np.full(n, 0)
    i0 = 0
    i1 = search_head(time, minutes)
    if i1 < 0:
        return rng, counts
    while True:
        d = array[i0: i1 + 1]
        rng[i1] = max(d) - min(d)
        counts[i1] = len(d)
        i1 += 1
        if i1 > n - 1:
            break
        index = -1
        for i in range(i0, i1):
            if time[i] > time[i1] - 60 * minutes:
                index = i
                break
        i0 = i1 if index < 0 else index        
    return rng, counts

def explosion(time, value, term_minutes):
    STATE_NONE = 0
    STATE_UP = 1
    STATE_DOWN = -1
    
    state = STATE_NONE
    n = len(time)
    bo = np.full(n, 0)
    i0 = 0
    t = time[i0] + 60 * term_minutes
    i1 = search(time, 0, t)
    if i1 < 0:
        return sig
    vmax = None
    vmin = None
    while i1 < n:
        d = value[i0: i1]
        brk_new = False
        if state == STATE_NONE:
            if value[i1] > max(d):
                    bo[i1] = 1
                    vmax = value[i1]
                    state = STATE_UP
                    brk_new = True
            elif value[i1] < min(d):
                    bo[i1] = -1
                    vmin= value[i1]
                    state = STATE_DOWN
                    brk_new = True
        elif state == STATE_UP:
            if value[i1] < min(d):
                bo[i1] = -1
                vmin = value[i1]
                state = STATE_DOWN
                brk_new = True
            elif value[i1] > vmax:
                bo[i1] = 1
                vmax = value[i1]
        elif state == STATE_DOWN:
            if value[i1] > max(d):
                bo[i1] = 1
                vmax = value[i1]
                state = STATE_UP
                brk_new = True
            elif value[i1] < vmin:
                bo[i1] = -1
                vmin = value[i1]
        i1 += 1
        if i1 >= n:
            break
        if not brk_new:
            t = time[i1] - 60 * term_minutes
            i0 = search_reverse(time, i1, t)
    return bo

def probability(time, vector, term_minutes):
    n = len(vector)
    out = np.full(n, 0)
    i0 = 0
    i1 = search_head(time, term_minutes)
    while True:
        d = vector[i0: i1 + 1]
        out[i1] = sum(d)
        i1 += 1
        if i1 > n - 1:
            break
        index = -1
        for i in range(i0, i1):
            t = time[i1] - 60 * term_minutes
            if time[i] > t:
                index = i
                break
        i0 = i1 if index < 0 else index
    return out

def make_signal(values, threshold):
    array = slice_data(values, threshold)
    n = len(array)
    entry = np.full(n, 0)
    ext  = np.full(n, 0)
    sig = np.full(n, np.nan)
    state = 0
    for i in range(n):
        if state == 0:
            if array[i] != 0:
                entry[i] = array[i]
                sig[i] = array[i]
        elif state == 1:
            if array[i] == 0:
                ext[i] = 1
                sig[i] = 0
            elif array[i] == -1:
                ext[i] = 1
                entry[i] = -1
                sig[i] = -1
        elif state == -1:
            if array[i] == 0:
                ext[i] = 1
                sig[i] = 0
            elif array[i] == 1:
                entry[i] = 1
                ext[i] = 1
                sig[i] = 1
        state = array[i]
    return entry, ext, sig
    
def slice_data(array, value):
    n = len(array)
    out = np.full(n, 0)
    for i in range(n):
        if array[i] > 0:
            if array[i] >= value:
                out[i] = 1
        else:
            if array[i] <= -value:
                out[i] = -1
    return out

def separate(array, signal, values):
    n = len(array)
    out = []
    for value in values:
        a = np.full(n, np.nan)
        for i in range(n):
            if signal[i] == value:
                a[i] = array[i]
        out.append(a)
    return out
 
def trailing(signal, price, sl, trail_target, trail_stop):
    n = len(price)
    SL = 1
    TRAIL_STOP = 2
    TIME_END = 0
    fired = False
    position_max = None
    for i in range(n):
        if signal == 1:
            profit = price[i] - price[0]
            if profit < -sl:
                return(i, profit, SL)
        elif signal == -1:
            profit = price[0] - price[i]
            if profit < -sl:
                return(i, profit, SL)
        if trail_target == 0 or trail_stop == 0:
            continue
        if signal == 1:
            profit = price[i] - price[0]
        elif signal == -1:
            profit = price[0] - price[i]
        if fired:
            if profit > profit_max:
                profit_max = profit
            elif (profit - profit_max) < -trail_stop:
                return(i, profit, TRAIL_STOP)
        else:
            if profit >= trail_target:
                fired = True
                profit_max = profit
    if signal == 1:
        profit = price[n - 1] - price[0]
    elif signal == -1:
        profit = price[0] - price[n - 1]
    return (n - 1, profit, TIME_END)
    
 
def evaluate(time, price, signal, trade_param):
    sl = trade_param['sl']
    trail_target = trade_param['trail_target']
    trail_stop = trade_param['trail_stop']
    def search_exit(index): 
        n = len(time)
        for i in range(index, n):
                if signal[i] == 0:
                    return i
        return -1
    
    positions = []
    n = len(time)
    i = 0
    s = 0
    times = []
    profits = []
    while i < n:
        if signal[i] == 1 or signal[i] == -1:
            j = search_exit(i + 1)
            if j < 0:
                break
            index, profit, reason = trailing(signal[i], price[i + 1: j + 2], sl, trail_target, trail_stop)
            index += i + 1
            s += profit        
            times.append(time[j + 1])
            profits.append(s)
            positions.append([signal[i], time[i + 1], price[i + 1], time[j + 1], price[j + 1], profit, reason])
            i = j + 2
        else:
            i += 1
    return (times, profits), positions, s

def entries(time, positions):
    out = []
    tbegin = time[0]
    tend = time[-1]
    for p in positions:
        signal, t0, p0, t1, p1, profit, reason = p
        if t0 >= tbegin and t0 <= tend and t1 >= tbegin and t1 <= tend:
            i0 = search(time, 0, t0)
            if i0 < 0:
                continue
            i1 = search(time, i0, t1)
            if i1 < 0:
                continue
            out.append((signal, i0, i1, profit, reason))
    return out

def simulate(optimize, number, symbol, dirpath, data0, technical_param, trade_param, hours=6):
    breskout_minutes = technical_param['breakout_minutes']
    prob_minutes = technical_param['prob_minutes']
    prob_threshold = technical_param['prob_threshold']
    jst0 = data0[Columns.JST]
    time = data0['timestamp']
    price = data0['ask']
    rng, counts = range_minutes(time, price, 60)
    bo = explosion(time, price, breskout_minutes)
    norm, up, down = separate(price, bo, [0, 1, -1])
    prob = probability(time, bo, prob_minutes)
    entry, ext, sig = make_signal(prob, prob_threshold)
    profits, positions, s = evaluate(jst0, price, sig, trade_param)
    print('#', number, 'profit:', s)

    if (optimize and s > 300) or (not optimize):
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        axes[0].plot(profits[0], profits[1], color='red', alpha=0.5)
        ax0 = axes[0].twinx()
        ax0.scatter(jst0, price, color='blue', alpha=0.01)
        axes[1].plot(jst0, rng, color='blue', alpha=0.5, label='Range')
        ax1 = axes[1].twinx()
        ax1.plot(jst0, counts, color='red', alpha=0.5, label='Counts')
        ax1.legend()
        ax0.legend()
        [ax.legend() for ax in axes]
        fig.savefig(os.path.join(dirpath, f'{number}_profits.png'))
        df = pd.DataFrame(positions, columns=['signal', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 'reason'])
        df.to_csv(os.path.join(dirpath, f'{number}_positions.csv'), index=False)
    
    if optimize:
        return s
    
    data0['breakout'] = bo
    data0['range'] = rng
    data0['counts'] = counts
    data0['prob'] = prob
    data0['entry'] = entry
    data0['exit'] = ext
    data0['signal'] = sig  
    
    tbegin = jst0[0]
    tend = jst0[-1]
    year = tbegin.year
    month = tbegin.month
    day = tbegin.day
    
    t0 = datetime(year, month, day).astimezone(JST)
    t1 = t0 + timedelta(hours=hours)
    count = 0
    while t1 < tend:
        n, data = TimeUtils.slice(data0, 'jst', t0, t1)
        if n > 10:
            time = data['timestamp']
            jst = data['jst']
            price = data['ask']
            bo = data['breakout']
            signal = data['signal']
            prob = data['prob']
            entry = data['entry']
            ext = data['exit']
            rng = data['range']
            counts = data['counts']
            norm, up, down = separate(price, bo, [0, 1, -1])
            #evaluate(time, price, entry, ext, rng, counts)    
            fig, axes = plt.subplots(2, 1, figsize=(20, 10))
            axes[0].scatter(jst, norm, color='blue', alpha=0.01, s=5)
            axes[0].scatter(jst, up, color='green', alpha=0.1, s=20)
            axes[0].scatter(jst, down, color='red', alpha=0.1, s=20)
            ax0 = axes[0].twinx()
            ax0.plot(jst, prob, color='orange', alpha=0.5)
            ax0.plot(jst, np.array(bo) * 10, color='yellow', alpha=.7)
            ax0.plot(jst, np.array(signal) * 10, color='red', alpha=0.5)
            for sig, i0, i1, profit, reason in entries(jst, positions):
                if sig == 1:
                    color = 'green'
                    marker = '^'
                    value = max(price)
                else:
                    color = 'red'
                    marker = 'v'
                    value = min(price)
                axes[0].scatter(jst[i0], price[i0], color=color, marker=marker, s=200, alpha=0.5)
                axes[0].scatter(jst[i1], price[i1], color='gray', marker='X', s=200, alpha=0.5)
                #axes[0].vlines(jst[i1], ymin=min(price), ymax=max(price), color='black', alpha=0.8)
                if profit > 0:
                    color = 'green'
                else:
                    color = 'red'
                axes[0].text(jst[i1], value, f'{profit:.2f}', color=color)  
            for i, v in enumerate(ext):
                if v == 1:
                    axes[0].vlines(jst[i], ymin=min(price), ymax=max(price), color='black', alpha=0.5)
            axes[1].plot(jst, rng, color='red', alpha=0.5, label='Range')
            ax1 = axes[1].twinx()
            ax1.plot(jst, counts, color='blue', alpha=0.5, label='Counts')
            ax1.legend()
            ax1.set_ylabel('Counts')
            axes[1].legend()
            axes[1].set_ylabel('Range')
            fig.savefig(os.path.join(dirpath, f"#{count}.png"))
            plt.close()
            count += 1
        t0 = t1
        t1 = t0 + timedelta(hours=hours)
        
def make_technical_param(randomize=True):
    if randomize:
        breakout_minutes = 5 * random.randint(2, 20)
        prob_minutes = 2 * random.randint(1, 30)
        prob_threshold = 5 * random.randint(1, 10)
    else:
        breakout_minutes = 30
        prob_minutes = 30
        prob_threshold = 5
    return {'breakout_minutes': breakout_minutes, 'prob_minutes': prob_minutes, 'prob_threshold': prob_threshold}

def make_trade_param(randomize=True):
    if randomize:
        sl = 5 * random.randint(1, 100)
        trail_target = 5 * random.randint(1, 100)
        trail_stop = 5 * random.randint(1, 100)
    else:
        sl = 30
        trail_target = 50
        trail_stop = 20
    return {'sl': sl, 'trail_target': trail_target, 'trail_stop': trail_stop}

def optimize(symbol, id, repeat=1000):
    dirpath = f'./optimize/breakout_tick_2024.12/{symbol}/{id}'
    os.makedirs(dirpath, exist_ok=True)
    data0 = from_pickle(symbol)
    out = []
    for i in range(repeat):
        technical_param = make_technical_param()
        trade_param = make_trade_param()
        s = simulate(True, i, symbol, dirpath, data0.copy(), technical_param, trade_param)
        p1, columns1 = expand('p1', technical_param)
        p2, columns2 = expand('p2', trade_param)
        out.append([i] + p1 + p2 + [s])
        try:
            df = pd.DataFrame(out, columns=['id'] + columns1 + columns2 + ['profit'])
            df.to_csv(os.path.join(dirpath, 'optimize.csv'), index=False)
        except:
            pass

if __name__ == '__main__':
    args = sys.argv
    if len(args) == 3:    
        symbol = args[1]
        id = args[2]
    else:
        symbol = 'NIKKEI'
        id = 'a'
    print(symbol)
    optimize(symbol, id)