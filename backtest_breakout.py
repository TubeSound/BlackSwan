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
    filepath = './data/Axiory/tick/NSDQ_TICK_2025_01_15.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
        jst = list(data0['jst'])
        bid = list(data0['bid'])
        ask = list(data0['ask'])
        dic = {Columns.JST: jst, Columns.BID: bid, Columns.ASK: ask}
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

def explosion(time, value, term_minutes):
    STATE_NONE = 0
    STATE_UP = 1
    STATE_DOWN = -1
    
    state = STATE_NONE
    n = len(time)
    sig = np.full(n, 0)
    bo = np.full(n, 0)
    i0 = 0
    t = time[i0] + timedelta(minutes=term_minutes)
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
            t = time[i1] - timedelta(minutes=term_minutes)
            i0 = search_reverse(time, i1, t)
    return sig, bo

def majority(vector, term, rate):
    n = len(vector)
    out = np.full(n, 0)
    for i in range(term - 1, n):
        d = vector[i - term + 1: i + 1]
        mean = sum(d) / len(d)
        if abs(mean) > rate:
            if mean > 0:
                out[i] = 1
            else:
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
 
def evaluate(symbol, strategy, hours=4):
    dirpath = './debug'
    os.makedirs(dirpath, exist_ok=True)
    data0 = from_pickle(symbol)
    jst = data0[Columns.JST]
    tbegin = jst[0]
    year = tbegin.year
    month = tbegin.month
    day = tbegin.day
    
    t0 = datetime(year, month, day).astimezone(JST)
    t1 = t0 + timedelta(hours=hours)
    count = 0
    while t1 < jst[-1]:
        n, data = TimeUtils.slice(data0, jst, t0, t1)
        if n > 10:
            time = data['jst']
            price = data['ask']
            sig, bo = explosion(time, price, 60)
            norm, up, down = separate(price, bo, [0, 1, -1])
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.scatter(time, norm, color='blue', alpha=0.01, s=5)
            ax.scatter(time, up, color='green', alpha=0.2, s=50)
            ax.scatter(time, down, color='red', alpha=0.2, s=50)
            ax2 = ax.twinx()
            ax2.plot(time, bo, color='orange', alpha=0.5)
            ax2.set_ylim(-2, 2)
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