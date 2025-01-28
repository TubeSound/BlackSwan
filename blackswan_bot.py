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
from technical import  ATRP, MA
from common import Signal, Indicators
from line_notify import LineNotify

CMAP = plt.get_cmap("tab10")
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  

INITIAL_DATA_LENGTH = 24 * 240



# -----

scheduler = sched.scheduler()


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


class BlackSwanBot():
    def __init__(self, symbols:[str], timeframe:str='H1', interval_seconds:int=60 * 10):
    #def __init__(self, symbols:[str], timeframe:str='M5', interval_seconds:int=60):
        self.symbols = symbols
        self.timeframe = timeframe
        self.invterval_seconds = interval_seconds
        self.notify = LineNotify() 
        self.delta_hour_from_gmt = None
        self.server_timezone = None
        self.page = 0
        self.buffers = {}


    def calc_indicators(self, timeframe, data: dict, params):
        ATRP(data, 40, ma_window=40)
        MA(data, 4 * 24 * 2, 4 * 8)
        
        
    def set_sever_time(self, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        print('SeverTime GMT+', dt, tz)
        

    def run(self):
        for i, symbol in enumerate(self.symbols):
            mt5 = Mt5Trade(symbol)
            
            df = mt5.get_rates(self.timeframe, INITIAL_DATA_LENGTH)
            if len(df) < INITIAL_DATA_LENGTH:
                raise Exception('Error in initial data loading')
            ret =  is_market_open(mt5, self.server_timezone)
            if ret:
                # last data is invalid
                df = df.iloc[:-1, :]
            else:
                if i == 0:
                    print('<マーケットクローズ>')
            
            buffer = DataBuffer(self.calc_indicators, symbol, self.timeframe, df, {}, self.delta_hour_from_gmt)
            self.buffers[symbol] = buffer
            os.makedirs('./debug', exist_ok=True)
            #save(buffer.data, './debug/initial_' + self.symbol + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.xlsx')
        #self.save_chart(24 * 30 * 6)
        return ret
        
        
    def update(self):
        alerts = []
        for symbol in self.symbols:
            mt5 = Mt5Trade(symbol)
            df = mt5.get_rates(self.timeframe, 2)
            df = df.iloc[:-1, :]
            buffer = self.buffers[symbol]
            n = buffer.update(df)
            if n > 0:
                current_time = buffer.last_time()
                current_index = buffer.last_index()
                atrp = buffer.data[Indicators.ATRP][-1]
                if atrp > 0.5:
                    alerts.append(symbol)
                print(symbol, 'ATRP:', atrp)
                time.sleep(10)
        if len(alerts) > 0:
            path = self.save_chart(INITIAL_DATA_LENGTH)
            self.notify.send(f'{alerts} 羽ばたきました', image=path)
        return n   
        
        
    def save_chart(self, length):
        fig, axes = gridFig([5, 3], (10, 8))
        i = 0
        for symbol, buffer in self.buffers.items():
            data = buffer.data            
            jst = data['jst'][-length:]
            cl = data['close'][-length:]
            atrp = data['ATRP'][-length:]
            ma = data['MA_LONG'][-length:]
            if symbol == 'NIKKEI':
                axes[0].plot(jst, cl, label=symbol, color='blue', alpha=0.5)
                axes[0].plot(jst, ma, label='MA', color='purple')
            axes[1].plot(jst, atrp, color=CMAP(i), label=symbol, alpha=0.95)
            axes[1].hlines(0.5, jst[0], jst[-1], color='yellow', linewidth=2.0)
            i += 1
        [ax.grid() for ax in axes]
        [ax.legend() for ax in axes]
        [ax.set_xlim(jst[0], jst[-1]) for ax in axes]
        axes[1].set_ylim(0, 2.0)
        axes[1].set_title('ATRP')
        axes[0].set_title('NIKKEI '  + self.timeframe)
        dirpath = './tmp/chart/'
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, f'ATRP(H1).png')
        fig.savefig(path)
        return path

def create_bot():
    symbols =  ['NIKKEI', 'DOW', 'SP', 'FTSE', 'DAX', 'USDJPY', 'XAUUSD']
    bot = BlackSwanBot(symbols)    
    bot.set_sever_time(3, 2, 11, 1, 3.0)
    return bot


def test():
    bot = create_bot()
    Mt5Trade.connect()
    bot.run()
    while True:
        scheduler.enter(10, 1, bot.update)
        scheduler.run()

if __name__ == '__main__':
    test()