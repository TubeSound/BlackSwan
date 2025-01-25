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
    filepath = f'./data/Axiory/tick/{symbol}_TICK_2025_01_10.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
        jst = list(data0['jst'])
        bid = list(data0['bid'])
        ask = list(data0['ask'])
        timestamp = [t.timestamp() for t in jst]
        dic = {Columns.JST: jst, Columns.TIMESTAMP: timestamp, Columns.BID: bid, Columns.ASK: ask}
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
    sig = np.full(n, 0)
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
                    sig[i1] = 1
                    bo[i1] = 1
                    vmax = value[i1]
                    state = STATE_UP
                    brk_new = True
            elif value[i1] < min(d):
                    sig[i1] = -1
                    bo[i1] = -1
                    vmin= value[i1]
                    state = STATE_DOWN
                    brk_new = True

        elif state == STATE_UP:
            if value[i1] < min(d):
                sig[i1] = -1
                bo[i1] = -1
                vmin = value[i1]
                state = STATE_DOWN
                brk_new = True
            elif value[i1] > vmax:
                bo[i1] = 1
                vmax = value[i1]
        elif state == STATE_DOWN:
            if value[i1] > max(d):
                sig[i1] = 1
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
    return sig, bo

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
 
def evaluate(symbol, strategy, hours=6):
    dirpath = f'./debug/{symbol}'
    os.makedirs(dirpath, exist_ok=True)
    data0 = from_pickle(symbol)
    jst0 = data0[Columns.JST]
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
            rng, counts = range_minutes(time, price, 5)
            sig, bo = explosion(time, price, 30)
            norm, up, down = separate(price, bo, [0, 1, -1])
            prob = probability(time, bo, 15)
            entry, ext, sig = make_signal(prob, 10)
            
            fig, axes = plt.subplots(2, 1, figsize=(20, 10))
            axes[0].scatter(jst, norm, color='blue', alpha=0.01, s=5)
            axes[0].scatter(jst, up, color='green', alpha=0.1, s=20)
            axes[0].scatter(jst, down, color='red', alpha=0.1, s=20)
            ax0 = axes[0].twinx()
            ax0.plot(jst, prob, color='orange', alpha=0.5)
            ax0.plot(jst, np.array(bo) * 10, color='yellow', alpha=.7)
            ax0.plot(jst, np.array(sig) * 10, color='red', alpha=0.5)
            for i, e in enumerate(entry):
                if e != 0:
                    if e > 0:
                        color = 'green'
                        marker = '^'
                    else:
                        color = 'red'
                        marker = 'v'
                    axes[0].scatter(jst[i], price[i], color=color, marker=marker, s=200, alpha=0.5)
                    axes[0].vlines(jst[i], ymin=min(price), ymax=max(price), color=color, alpha=0.5)
            for i, e in enumerate(ext):
                if e != 0:
                    axes[0].scatter(jst[i], price[i], color='gray', marker='X', s=200, alpha=0.5)
                    axes[0].vlines(jst[i], ymin=min(price), ymax=max(price), color='black', alpha=0.8)  
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

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        symbol = 'NSDQ'
        strategy = 'breakout'
    else:        
        symbol = args[1]
        strategy = args[2]
    print(symbol, strategy)
    evaluate(symbol, strategy)