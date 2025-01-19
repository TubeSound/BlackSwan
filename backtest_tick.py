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
        jst = data0['jst']
        bid = data0['bid']
        ask = data0['ask']
        dic = {Columns.JST: jst, Columns.BID: bid, Columns.ASK: ask}
        return dic        
    return None




def timefilter(data, year_from, month_from, day_from, year_to, month_to, day_to):
    t0 = datetime(year_from, month_from, day_from).astimezone(JST)
    t1 = datetime(year_to, month_to, day_to).astimezone(JST)
    jst = data[Columns.JST]
    return TimeUtils.slice(data, jst, t0, t1)


 
    
    
    
    
    
 
 
def evaluate(symbol, strategy):
    data0 = from_pickle(symbol)
    jst = data[Columns.JST]
    tbegin = jst[0]
    year = tbegin.year()
    month = tbegin.month()
    day = tbegin.day()
    
    t0 = datetime(year, month, day).astimezone(JST)
    t1 = t0 + timedelta(days=1)
    while t1 < jst[-1:]:
        data = TimeUtils.slice(data0, jst, t0, t1)
        time = data['jst']
        price = data['ask']
        
    
 

    
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