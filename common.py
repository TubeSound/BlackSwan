# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:59:00 2023

@author: docs9
"""
import MetaTrader5 as mt5


HOLD = 0
DOWN = -1
UP = 1
DOWN_TO_UP = 2
UP_TO_DOWN = 3
LOW = -1
HIGH = 1

class Columns:
    TIME = 'time'
    JST = 'jst'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'    
    ASK = 'ask'
    BID = 'bid'
    MID = 'mid'
    VOLUME = 'volume'

class TimeFrame:
    TICK = 'TICK'
    M1 = 'M1'
    M5 = 'M5'
    M15 = 'M15'
    M30 = 'M30'
    H1 = 'H1'
    H4 = 'H4'
    D1 = 'D1'
    W1 = 'W1'
    
    timeframes = {  M1: mt5.TIMEFRAME_M1, 
                    M5: mt5.TIMEFRAME_M5,
                    M15: mt5.TIMEFRAME_M15,
                    M30: mt5.TIMEFRAME_M30,
                    H1: mt5.TIMEFRAME_H1,
                    H4: mt5.TIMEFRAME_H4,
                    D1: mt5.TIMEFRAME_D1,
                    W1: mt5.TIMEFRAME_W1}
            
    @staticmethod 
    def const(timeframe_str: str):
        return TimeFrame.timeframes[timeframe_str]

class Indicators:
    MA = 'MA'
    MA_SHORT = 'MA_SHORT'
    MA_MID = 'MA_MID'
    MA_LONG = 'MA_LONG'
    MA_LONG_HIGH = 'MA_LONG_HIGH'
    MA_LONG_LOW = 'MA_LONG_LOW'
    
    MA_LONG_SLOPE = 'MA_LONG_SLOPE'
    MA_MID_SLOPE = 'MA_MID_SLOPE'
    MA_SHORT_SLOPE = 'MA_SHORT_SLOPE'

    MABAND = 'MABAND'
    MABAND_LONG = 'MABAND_LONG'
    MABAND_SHORT = 'MABAND_SHORT'    
    
    MAGAP = 'MAGAP'
    MAGAP_SLOPE= 'MAGAP_SLOPE'
    MAGAP_ENTRY = 'MAGAP_ENTRY'
    MAGAP_EXIT = 'MAGAP_EXIT'
    
    TR = 'TR'
    ATR = 'ATR'
    ATR_LONG = 'ATR_LONG'
    ATR_UPPER = 'ATR_UPPER'
    ATR_LOWER = 'ATR_LOWER'
    ATRP = 'ATRP'
    DX = 'DX'
    ADX = 'ADX'
    ADX_LONG = 'ADX_LONG'
    DI_PLUS = 'DI_PLUS'
    DI_MINUS = 'DI_MINUS'
    POLARITY = 'POLARITY'
    
    ATR_TRAIL = 'ATR_TRAIL'
    ATR_TRAIL_SIGNAL = 'ATR_TRAIL_SIGNAL'
    ATR_TRAIL_U = 'ATR_TRAIL_U'
    ATR_TRAIL_L = 'ATR_TRAIL_L'
    
    SUPERTREND_U = 'SUPERTREND_U'
    SUPERTREND_L = 'SUPERTREND_L'
    SUPERTREND = 'SUPERTREND'
    SUPERTREND_SIGNAL = 'SUPERTREND_SIGNAL'
    SUPERTREND_STOP_PRICE = 'SUPERTREND_STOP_PRICE'
    
    BB = 'BB'
    BB_MA = 'BB_MA'
    BB_UPPER = 'BB_UPPER'
    BB_LOWER = 'BB_LOWER'
    BB_UP = 'BB_UP'
    BB_DOWN = 'BB_DOWN'
    BB_CROSS = 'BB_CROSS'
    BB_CROSS_UP = 'BB_CROSS_UP'
    BB_CROSS_DOWN = 'BB_CROSS_DOWN'
    
    TREND_ADX_DI ='TREND_ADX_DI'
    
    BBRATE = 'BBRATE'
    VWAP = 'VWAP'
    VWAP_RATE = 'VWAP_RATE'
    VWAP_SLOPE = 'VWAP_SLOPE'
    VWAP_U = 'VWAP_U'
    VWAP_L = 'VWAP_L'
    VWAP_PROB = 'VWAP_PROB'
    VWAP_DOWN = 'VWAP_DOWN'
    VWAP_CROSS_DOWN = 'VWAP_CROSS_DOWN'
    VWAP_RATE_SIGNAL = 'VWAP_RATE_SIGNAL'
    VWAP_PROB_SIGNAL = 'VWAP_PROB_SIGNAL'
    
    RCI = 'RCI'
    RCI_SIGNAL = 'RCI_SIGNAL'
    
    FILTER_MA = 'FILTER_MA'
    
    PPP_ENTRY = 'PPP_ENTRY'
    PPP_EXIT = 'PPP_EXIT'
    
    MA_GOLDEN_CROSS = 'MA_GOLDEN_CROSS'

class Signal:
    LONG = 1
    SHORT = -1    