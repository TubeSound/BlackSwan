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
from technical import PPP
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

INITIAL_DATA_LENGTH = 12 * 48

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
        self.notify = LineNotify() 
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
        ppp = param['PPP']
        long_term = ppp['long_term']
        mid_term = ppp['mid_term']
        short_term = ppp['short_term']
        tap = ppp['tap']
        threshold = ppp['threshold']
        pre = 12 * 4
        post = 12 * 1
        target = 12 * 4
        PPP(timeframe, data, long_term, mid_term, short_term, pre, post, target, threshold=threshold, tap=tap)
        
        
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
    
        jst = data[Columns.JST]
        n = len(jst)
        
        jst = jst[n-length:]
        op = data[Columns.OPEN][n-length:]
        hi = data[Columns.HIGH][n-length:]
        lo = data[Columns.LOW][n-length:]
        cl = data[Columns.CLOSE][n-length:]
        ma_l = data[Indicators.MA_LONG][n-length:]
        ma_m = data[Indicators.MA_MID][n-length:]
        ma_s = data[Indicators.MA_SHORT][n-length:]
        up = data[Indicators.PPP_UP][n - length:]
        down = -1 * np.array(data[Indicators.PPP_DOWN][n - length:])
        fig, axes = gridFig([5, 1], (8, 5))
        chart = CandleChart(fig, axes[0], title=title, date_format=CandleChart.DATE_FORMAT_TIME)
        chart.drawCandle(jst, op, hi, lo, cl)
        chart.drawLine(jst, ma_l, color='blue')
        chart.drawLine(jst, ma_m, color='green')
        chart.drawLine(jst, ma_s, color='red')
        chart2 = CandleChart(fig, axes[1], title=title, date_format=CandleChart.DATE_FORMAT_TIME)
        chart2.drawLine(jst, up, color='green', alpha=0.5)
        chart2.drawLine(jst, down, color='red', alpha=0.5)
        chart2.ylimit((-2, 2))
        dirpath = './tmp/chart/'
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, f'chart_{title}.png')
        fig.savefig(path)
        return path
        
        

def technical_param(symbol):
    param_nikkei = {'PPP': {
                        'long_term': 60,
                        'mid_term': 20,
                        'short_term': 5,
                        'tap': 0,
                        'threshold': 0.01
                    }
            }
    if symbol.lower() == 'nikkei':
        return param_nikkei
    
    param_dow = {'PPP': {
                        'long_term': 60,
                        'mid_term': 20,
                        'short_term': 5,
                        'tap': 0,
                        'threshold': 0.01
                    }
            }
    if symbol.lower() == 'dow':
        return param_dow
    
    

def create_bot(symbol, timeframe):
    bot = Bot(symbol, timeframe, 1, Indicators.PPP_ENTRY, Indicators.PPP_EXIT, technical_param(symbol))    
    bot.set_sever_time(3, 2, 11, 1, 3.0)
    return bot

def create_usdjpy_bot():
    symbol = 'USDJPY'
    timeframe = 'M5'
    technical = {'atr_window': 40, 'atr_multiply': 3.0}
    trade = {'sl': 0.3, 'target_profit': 0.4, 'trailing_stop': 0.1, 'volume': 0.1, 'position_max': 5, 'timelimit': 40}
    bot = Bot(symbol, timeframe, 1, Indicators.SUPERTREND_SIGNAL, technical, trade)    
    return bot
     
def test():
    
    bot1 = create_bot( 'NIKKEI', 'M5')
    Mt5Trade.connect()
    bot1.run()
    bot2 = create_bot('DOW', 'M5')
    bot2.run()
    while True:
        scheduler.enter(10, 1, bot1.update)
        scheduler.enter(10, 2, bot2.update)
        scheduler.run()

if __name__ == '__main__':
    test()