import os
import sys
sys.path.append('../Libraries/trade')
import pickle

import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
from mt5_trade import Mt5Trade, Columns, PositionInfo
import sched

from candle_chart import CandleChart, makeFig, gridFig
from data_buffer import DataBuffer
from time_utils import TimeUtils
from utils import Utils
from technical import SUPERTREND, SUPERTREND_SIGNAL
from strategy import Simulation
from common import Signal, Indicators

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

INITIAL_DATA_LENGTH = 500

# -----

scheduler = sched.scheduler()

# -----
def utcnow():
    #utc1 = datetime.utcnow()
    #utc1 = utc1.replace(tzinfo=UTC)
    utc = datetime.now(UTC)
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
    
class TradeManager:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.positions = {}
        self.positions_closed = {}

    def add_position(self, position: PositionInfo):
        self.positions[position.ticket] = position
       
    def move_to_closed(self, ticket):
        if ticket in self.positions.keys():
            pos = self.positions.pop(ticket)
            self.positions_closed[ticket] = pos

        else:
            print('move_to_closed, No tickt')
        
    def summary(self):
        out = []
        for ticket, pos in list(self.positions.items()) + list(self.positions_closed.items()):
            d, columns = pos.array()
            out.append(d)
        df = pd.DataFrame(data=out, columns=columns)
        return df

    def remove_positions(self, tickets):
        for ticket in tickets:
            self.move_to_closed(ticket)
            
    def open_positions(self):
        return self.positions
    
    def untrail_positions(self):
        positions = {}
        for ticket, position in self.positions.items():
            if position.profit_max is None:
                positions[ticket] = position
        return positions
    
    def remove_position_auto(self, mt5_positions):
        remove_tickets = []
        for ticket, info in self.positions.items():
            found = False
            for position in mt5_positions:
                if position.ticket == ticket:
                    found = True
                    break    
            if found == False:
                remove_tickets.append(ticket)
        if len(remove_tickets):
            self.remove_positions(remove_tickets)    
            print('<Closed by Meta Trader Stoploss or Takeprofit> ', self.symbol, 'tickets:', remove_tickets)
            
      
class TradeBot:
    def __init__(self, symbol:str,
                 timeframe:str,
                 interval_seconds:int,
                 entry_column: str,
                 exit_column:str, 
                 technical_param: dict,
                 trade_param:dict,
                 simulate=False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.invterval_seconds = interval_seconds
        self.entry_column = entry_column
        self.exit_column = exit_column
        self.technical_param = technical_param
        self.trade_param = trade_param
        if not simulate:
            mt5 = Mt5Trade(symbol)
            self.mt5 = mt5
        self.delta_hour_from_gmt = None
        self.server_timezone = None
        
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
        p = param
        SUPERTREND(data, p['atr_window'], p['atr_multiply'])
        SUPERTREND_SIGNAL(data, p['short_term'])
       
        
        
    def set_sever_time(self, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        print('SeverTime GMT+', dt, tz)
        
        
    def backup_path(self):
        dir_path = f'./trading/tmp/{self.symbol}/{self.timeframe}'
        path = os.path.join(dir_path, f'{self.symbol}_{self.timeframe}_trade_manager.pkl')
        return path, dir_path
        
    def load_trade_manager(self):
        path, dir_path = self.backup_path()
        try:
            with open(path, 'rb') as f:
                self.trade_manager = pickle.load(f)
            print(self.symbol, self.timeframe, ' loaded Trade_manager positions num: ', len(self.trade_manager.positions))
        except:
            self.trade_manager = TradeManager(self.symbol, self.timeframe)
            
    def save_trade_manager(self):
        path, dir_path = self.backup_path()
        os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.trade_manager, f)     
            
    def run(self):
        self.load_trade_manager()
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
            print(f'*マーケットクローズ: {self.symbol}')
            buffer = DataBuffer(self.calc_indicators, self.symbol, self.timeframe, df, self.technical_param, self.delta_hour_from_gmt)
            self.buffer = buffer
            return False
    
    def update(self):
        self.remove_closed_positions()
        record = self.trailing()
        df = self.mt5.get_rates(self.timeframe, 2)
        df = df.iloc[:-1, :]
        n = self.buffer.update(df)
        if n > 0:
            current_time = self.buffer.last_time()
            current_index = self.buffer.last_index()
            #save(self.buffer.data, './debug/update_' + self.symbol + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.xlsx')
            #self.check_timeup(current_index)
            entry_signal = self.buffer.data[self.entry_column][-1]
            exit_signal = self.buffer.data[self.exit_column][-1]
            if exit_signal != 0:
                positions = self.trade_manager.open_positions()
                if len(positions) > 0:
                    self.close_positions(positions, exit_signal)
                    self.debug_print('<Exit> position num:', len(positions))
                    record = True
            if entry_signal == Signal.LONG or entry_signal == Signal.SHORT:
                self.debug_print('<Signal> ', entry_signal)
                self.entry(self.buffer.data, entry_signal, current_index, current_time)
            if record:
                try:
                    df = self.trade_manager.summary()
                    dir_path = f'./trading/summary/{self.symbol}/{self.timeframe}'
                    os.makedirs(dir_path, exist_ok=True)
                    path = os.path.join(dir_path, f'{self.symbol}_{self.timeframe}_trade_summary.xlsx')
                    df.to_excel(path, index=False)
                except:
                    pass
            self.save_trade_manager()
        return n
    
            
                
    def mt5_position_num(self):
        positions = self.mt5.get_positions()
        count = 0
        for position in positions:
            if position.symbol == self.symbol:
                count += 1
        return count
        
    def entry(self, data, signal, index, time):
        volume = self.trade_param['volume']
        sl = self.trade_param['sl_value']
        trail_target = self.trade_param['trail_target']
        trailing_stop = self.trade_param['trail_stop']          
        timelimit = self.trade_param['timelimit']                       
        position_max = int(self.trade_param['position_max'])
        num =  self.mt5_position_num()
        if num >= position_max:
            self.debug_print('<Entry> Request Canceled ', self.symbol, index, time,  'Position num', num)
            return
        if sl == 0:
            atr = data[Indicators.ATR]
            sl = atr[index] * 2.0
            
        try:
            ret, position_info = self.mt5.entry(signal, index, time, volume, stoploss=sl, takeprofit=None)
            if ret:
                position_info.trail_target = trail_target
                self.trade_manager.add_position(position_info)
                self.debug_print('<Entry> signal', position_info.signal, position_info.symbol, position_info.entry_index, position_info.entry_time)
        except Exception as e:
            print(' ... Entry Error', e)
            
    def remove_closed_positions(self):
        positions = self.mt5.get_positions()
        self.trade_manager.remove_position_auto(positions)
        
    def trailing(self):
        trailing_stop = self.trade_param['trail_stop'] 
        trail_target = self.trade_param['trail_target']
        if trailing_stop == 0 or trail_target == 0:
            return
        remove_tickets = []
        for ticket, info in self.trade_manager.positions.items():
            price = self.mt5.current_price(info.signal())
            triggered, profit, profit_max = info.update_profit(price)
            if triggered:
                self.debug_print('<Trailing fired> profit', profit_max)
            if profit_max is None:
                continue
            if (profit_max - profit) > trailing_stop:
                ret, info = self.mt5.close_by_position_info(info)
                if ret:
                    remove_tickets.append(info.ticket)
                    self.debug_print('<Closed trailing stop> Success', self.symbol, info.desc())
                else:
                    self.debug_print('<Closed trailin stop> Fail', self.symbol, info.desc())    
        for ticket in remove_tickets:
            self.trade_manager.move_to_closed(ticket)
        return len(remove_tickets) > 0
                
    def check_timeup(self, current_index: int):
        positions = self.mt5.get_positions()
        timelimit = int(self.trade_param['timelimit'])
        remove_tickets = []
        for position in positions:
            if position.ticket in self.positions_info.keys():
                info = self.positions_info[position.ticket]
                if (current_index - info.entry_index) >= timelimit:
                    ret, info = self.mt5.close_by_position_info(info)
                    if ret:
                        remove_tickets.append(position.ticket)
                        self.debug_print('<Closed Timeup> Success', self.symbol, info.desc())
                    else:
                        self.debug_print('<Closed Timeup> Fail', self.symbol, info.desc())                                      
        self.trade_manager.remove_positions(remove_tickets)
       
    def close_positions(self, positions, signal):   
        removed_tickets = []
        for ticket, position in positions.items():
            if position.signal() == signal:
                ret, _ = self.mt5.close_by_position_info(position)
                if ret:
                    removed_tickets.append(position.ticket)
                    self.debug_print('<Closed> Success', self.symbol, position.desc())
                else:
                    self.debug_print('<Closed> Fail', self.symbol, position.desc())           
        self.trade_manager.remove_positions(removed_tickets)


def load_params(symbol, timeframe, volume, position_max):
    path = f'./optimize2stage_2020-2024/supertrend/{symbol}/{timeframe}/trade_best_param.csv'
    df = pd.read_csv(path)
    technical = {}
    technical['atr_window'] = df['p1.atr_window'].to_list()[0]
    technical['atr_multiply'] = df['p1.atr_multiply'].to_list()[0]
    technical['ma_window'] = df['p1.ma_window'].to_list()[0]
    technical['short_term'] = df['p1.short_term'].to_list()[0]
    trade = {}
    trade['begin_hour'] = df['p2.begin_hour'].to_list()[0] 
    trade['begin_minute'] = df['p2.begin_minute'].to_list()[0]
    trade['hours'] = df['p2.hours'].to_list()[0]
    trade['sl_method'] = df['p2.sl_method'].to_list()[0]
    trade['sl_value'] = df['p2.sl_value'].to_list()[0]
    trade['volume'] = volume
    trade['position_max'] = position_max
    trade['trail_target'] = df['p2.trail_target'].to_list()[0]
    trade['trail_stop'] = df['p2.trail_stop'].to_list()[0]
    trade['timelimit'] = df['p2.timelimit'].to_list()[0]
    return technical, trade
    

def create_bot(symbol, timeframe):
    technical_param, trade_param = load_params(symbol, timeframe, 2, 2)
    bot = TradeBot(symbol, timeframe, 1, Indicators.SUPERTREND_ENTRY, Indicators.SUPERTREND_EXIT, technical_param, trade_param)    
    bot.set_sever_time(3, 2, 11, 1, 3.0)
    return bot

     
def test():
    symbols = ['DOW', 'NIKKEI', 'NSDQ'] #, 'XAUUSD', 'XPDUSD', 'XAGUSD'] #, 'NVDA'] #, 'TSLA']
    symbols = ['TSLA']
    bots = {}
    for i, symbol in enumerate(symbols):
        bot = create_bot(symbol, 'M15')
        if i == 0:
            Mt5Trade.connect()
        bot.run()
        bots[symbol ] = bot
    while True:
        for i, symbol in enumerate(symbols):
            scheduler.enter(10, 1 + 1, bots[symbol].update)
            scheduler.run()

if __name__ == '__main__':
    test()