import os
import sys
sys.path.append('../Libraries/trade')

import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
from mt5_trade import Mt5Trade, Columns, PositionInfo
import sched

import matplotlib.pyplot as plt
from candle_chart import CandleChart, makeFig, gridFig
from data_buffer import DataBuffer
from time_utils import TimeUtils
from utils import Utils
from technical import SUPERTREND, SUPERTREND_SIGNAL
from common import Signal, Indicators
from line_notify import LineNotify

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  

import logging
os.makedirs('./log', exist_ok=True)
log_path = './log/trade_' + datetime.now().strftime('%y%m%d_%H%M') + '.log'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p"
)

INITIAL_DATA_LENGTH = 24 * 60

# -----

scheduler = sched.scheduler()

page = 0

# -----
def utcnow():
    utc = datetime.utcnow()
    utc = utc.replace(tzinfo=UTC)
    return utc

def utc2localize(aware_utc_time, timezone):
    t = aware_utc_time.astimezone(timezone)
    return t

def is_market_open(mt5, timezone):
    now = utcnow()
    t = utc2localize(now, timezone)
    t -= timedelta(seconds=5)
    df = mt5.get_ticks_from(t, length=100)
    return (len(df) > 0)
        
def wait_market_open(mt5, timezone):
    while is_market_open(mt5, timezone) == False:
        time.sleep(5)

def save(data, path):
    d = data.copy()
    time = d[Columns.TIME] 
    d[Columns.TIME] = [str(t) for t in time]
    jst = d[Columns.JST]
    d[Columns.JST] = [str(t) for t in jst]
    df = pd.DataFrame(d)
    df.to_excel(path, index=False)
    

class Bot:
    def __init__(self, symbol:str,
                 timeframe:str,
                 interval_seconds:int,
                 entry_column: str,
                 exit_column:str, 
                 technical_param: dict):
        self.symbol = symbol
        self.timeframe = timeframe
        self.invterval_seconds = interval_seconds
        self.entry_column = entry_column
        self.exit_column = exit_column
        self.technical_param = technical_param
        self.notify = LineNotify('Duck') 
        mt5 = Mt5Trade(symbol)
        self.mt5 = mt5
        self.delta_hour_from_gmt = None
        self.server_timezone = None
        self.page = 0
        
    def debug_print(self, *args):
        utc = utcnow()
        jst = utc2localize(utc, JST)
        t_server = utc2localize(utc, self.server_timezone)  
        s = 'JST*' + jst.strftime('%Y-%m-%d_%H:%M:%S') + ' (ServerTime:' +  t_server.strftime('%Y-%m-%d_%H:%M:%S') +')'
        for arg in args:
            s += ' '
            s += str(arg) 
        print(s)    
        
    def calc_indicators(self, timeframe, data: dict, param: dict):
        
        SUPERTREND(data, param['atr_window'], param['atr_multiply'])
        SUPERTREND_SIGNAL(data, param['short_term'])
        
        
    def set_sever_time(self, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        print('SeverTime GMT+', dt, tz)
        
    def run(self):
        df = self.mt5.get_rates(self.timeframe, INITIAL_DATA_LENGTH)
        if len(df) < INITIAL_DATA_LENGTH:
            raise Exception('Error in initial data loading')
        if is_market_open(self.mt5, self.server_timezone):
            # last data is invalid
            df = df.iloc[:-1, :]
            buffer = DataBuffer(self.calc_indicators, self.symbol, self.timeframe, df, self.technical_param, self.delta_hour_from_gmt)
            self.buffer = buffer
            os.makedirs('./debug', exist_ok=True)
            #save(buffer.data, './debug/initial_' + self.symbol + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.xlsx')
            return True            
        else:
            print('<マーケットクローズ>')
            buffer = DataBuffer(self.calc_indicators, self.symbol, self.timeframe, df, self.technical_param, self.delta_hour_from_gmt)
            self.buffer = buffer
            return False
    
    def update(self):
        df = self.mt5.get_rates(self.timeframe, 2)
        df = df.iloc[:-1, :]
        n = self.buffer.update(df)
        if n > 0:
            current_time = self.buffer.last_time()
            current_index = self.buffer.last_index()
            entry_signal = self.buffer.data[self.entry_column][-1]
            exit_signal = self.buffer.data[self.exit_column][-1]
            if entry_signal != 0 or exit_signal != 0:
                path = self.save_chart(f'{self.symbol}_{self.timeframe}', self.buffer.data, 12 * 24)
                if exit_signal > 0:
                    pass
                    #self.notify.send(f'{self.symbol} 手仕舞ってね ', image=path)
                elif entry_signal == Signal.LONG:
                    self.notify.send(f'{self.symbol} 買ってよし', image=path)
                elif entry_signal == Signal.SHORT:
                    self.notify.send(f'{self.symbol} 売ってよし', image=path)
                    
                dirpath = './tmp/data/'
                os.makedirs(dirpath, exist_ok=True)
                path = os.path.join(dirpath, f'data_{self.page}')
                df = pd.DataFrame(self.buffer.data)
                df.to_csv(path, index=False)
                self.page += 1
                    
                    
        return n

    def save_chart(self, title, data, length):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
        jst = data[Columns.JST]
        n = len(jst)
        
        jst = jst[n-length:]
        op = data[Columns.OPEN][n-length:]
        hi = data[Columns.HIGH][n-length:]
        lo = data[Columns.LOW][n-length:]
        cl = data[Columns.CLOSE][n-length:]
        entries = data[Indicators.SUPERTREND_ENTRY][n - length:]
        exits = data[Indicators.SUPERTREND_EXIT]
        ma = data[Indicators.SUPERTREND_MA][n - length:]
        up = data[Indicators.SUPERTREND_U][n - length:]
        down = data[Indicators.SUPERTREND_L][n - length:]
        ax.plot(jst, cl, color='blue', alpha=0.2)
        ax.scatter(jst, up, alpha=0.6, color='green', marker='o', s= 5)
        ax.scatter(jst, down, alpha=0.4, color='orange', marker='o', s= 5)
        ax.scatter(jst, ma, alpha=0.4, color='red', marker='o', s= 5)
        for i, entry in enumerate(entries):
            if entry == 1:
                color = 'green'
            elif entry == -1:
                color= 'red'
            else:
                continue
            ax.vlines(jst[i], min(cl), max(cl), lw=2, linestyle='dotted', alpha=0.4, color=color)
        dirpath = './tmp/chart/'
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, f'chart_{title}.png')
        fig.savefig(path)
        return path
        
        

def technical_param(symbol):
    param = {}
    if symbol == 'NIKKEI':
        param['atr_window'] = 29
        param['atr_multiply'] = 2.7
        param['ma_window'] = 48
        param['short_term'] = 11
    elif symbol == 'DOW':
        param['atr_window'] = 79
        param['atr_multiply'] = 2.4
        param['ma_window'] = 38
        param['short_term'] = 17
    elif symbol == 'NSDQ':
        param['atr_window'] = 29
        param['atr_multiply'] = 1.5
        param['ma_window'] = 14
        param['short_term'] = 13
    elif symbol == 'XAUUSD':
        param['atr_window'] = 21
        param['atr_multiply'] = 3.2
        param['ma_window'] = 93
        param['short_term'] = 34
    return param
    

def create_bot(symbol, timeframe):
    bot = Bot(symbol, timeframe, 1, Indicators.SUPERTREND_ENTRY, Indicators.SUPERTREND_EXIT, technical_param(symbol))    
    bot.set_sever_time(3, 2, 11, 1, 3.0)
    return bot

     
def bot():
    symbols = ['NIKKEI', 'DOW', 'NSDQ']
    bots = {}
    for i, symbol in enumerate(symbols):
        bot = create_bot(symbol, 'M15')
        if i == 0:
            Mt5Trade.connect()
        bot.run()
        bots[symbol] = bot
        
    while True:
        for i, symbol in enumerate(symbols):
            bot = bots[symbol]
            scheduler.enter(10, i + 1, bot.update)
        scheduler.run()

def main() :
    bot()
    

if __name__ == '__main__':
    main():