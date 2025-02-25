import numpy as np 
import math
import statistics as stat
from scipy.stats import rankdata
from scipy.signal import find_peaks
from sklearn.cluster import KMeans 
from common import Indicators, Signal, Columns, UP, DOWN, HIGH, LOW, HOLD
from datetime import datetime, timedelta
from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

    
def nans(length):
    return [np.nan for _ in range(length)]

def full(length, value):
    return [value for _ in range(length)]

def is_nan(value):
    if value is None:
        return True
    return np.isnan(value)

def is_nans(values):
    if len(values) == 0:
        return True
    for value in values:
        if is_nan(value):
            return True
    return False

def sma(vector, window):
    window = int(window)
    n = len(vector)
    out = full(n, np.nan)
    ivalid = window- 1
    if ivalid < 0:
        return out
    for i in range(ivalid, n):
        d = vector[i - window + 1: i + 1]
        out[i] = stat.mean(d)
    return out

def ema(vector, window):
    window = int(window)
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    n = len(vector)
    out = full(n, np.nan)
    ivalid = window- 1
    if ivalid < 0:
        return out
    for i in range(ivalid, n):
        d = vector[i - window + 1: i + 1]
        out[i] = np.sum(d * weights)
    return out

def slopes(signal: list, window: int, minutes: int, tolerance=0.0):
    n = len(signal)
    out = full(n, np.nan)
    for i in range(window - 1, n):
        d = signal[i - window + 1: i + 1]
        if np.min(d) == 0:
            continue        
        m, offset = np.polyfit(range(window), d, 1)
        if abs(m) > tolerance:
            out[i] = m / np.mean(d[:3]) * 100.0 / (window * minutes)  * 60 * 24
    return out


def slope(vector):
    n = len(vector)
    d = np.array(vector)
    m, offset = np.polyfit(range(n), d, 1)
    return m, offset    
    
def subtract(signal1: list, signal2:list):
    n = len(signal1)
    if len(signal2) != n:
        raise Exception('dont match list size')
    out = nans(n)
    for i in range(n):
        if is_nan(signal1[i]) or is_nan(signal2[i]):
            continue
        out[i] = signal1[i] - signal2[i]
    return out


def linearity(signal: list, window: int):
    n = len(signal)
    out = nans(n)
    for i in range(window, n):
        data = signal[i - window + 1: i + 1]
        if is_nans(data):
            continue
        m, offset = np.polyfit(range(window), data, 1)
        e = 0
        for j, d in enumerate(data):
            estimate = m * j + offset
            e += pow(estimate - d, 2)
        error = np.sqrt(e) / window / data[0] * 100.0
        if error == 0:
            out[i] = 100.0
        else:
            out[i] = 1 / error
    return out
            
            
def true_range(high, low, cl):
    n = len(high)
    out = nans(n)
    ivalid = 1
    for i in range(ivalid, n):
        d = [ high[i] - low[i],
              abs(high[i] - cl[i - 1]),
              abs(low[i] - cl[i - 1])]
        out[i] = max(d)
    return out


def rci(vector, window):
    n = len(vector)
    out = nans(n)
    for i in range(window - 1, n):
        d = vector[i - window + 1: i + 1]
        if is_nans(d):
            continue
        r = rankdata(d, method='ordinal')
        s = 0
        for j in range(window):
            x = window - j
            y = window - r[j] + 1
            s += pow(x - y, 2)
        out[i] = (1 - 6 * s / ((pow(window, 3) - window))) * 100
    return out

def roi(vector:list):
    n = len(vector)
    out = nans(n)
    for i in range(1, n):
        if is_nan(vector[i - 1]) or is_nan(vector[i]):
            continue
        if vector[i - 1] == 0:
            out[i] = 0.0
        else:
            out[i] = (vector[i] - vector[i - 1]) / vector[i - 1] * 100.0
    return out

def pivot(up: list, down: list, threshold: float=7 , left_length: int=5, right_length: int=5):
    n = len(up)
    state = full(n, 0)
    for i in range(left_length + right_length, n):
        if up[i] + down[i] < 90:
            continue
        left = up[i - left_length - right_length: i - right_length]
        right = up[i - right_length: i + 1]        
        range_left = max(left) - min(left)
        if range_left < 2:
            if np.mean(left) < 20:
                if (np.mean(right) - np.mean(left)) > threshold:
                    state[i] = HIGH
            if np.mean(left) > 80:
                if (np.mean(right) - np.mean(left)) < -threshold:
                    state[i] = LOW
    return state

def cross_value(vector: list, value):
    n = len(vector)
    up = nans(n)
    down = nans(n)
    cross = full(n, HOLD)
    for i in range(1, n):
        if vector[i - 1] < value and vector[i] >= value:
            up[i] = 1
            cross[i] = UP
        elif vector[i - 1] > value and vector[i] <= value:
            down[i] = 1
            cross[i] = DOWN
    return up, down, cross

def median(vector, window):
    n = len(vector)
    out = nans(n)
    for i in range (window, n):
        d = vector[i - window: i + 1]
        if is_nans(d):
            continue
        med = np.median(d)
        out[i] = med
    return out
        
def band_position(data, lower, center, upper):
    n = len(data)
    pos = full(n, 0)
    for i in range(n):
        if is_nan(data[i]):
            continue 
        if data[i] > upper[i]:
            pos[i] = 2
        else:
            if data[i] > center[i]:
                pos[i] = 1
        if data[i] < lower[i]:
            pos[i] = -2
        else:
            if data[i] < center[i]:
                pos[i] = -1
    return pos

def probability(position, states, window):
    n = len(position)
    prob = full(n, 0)
    for i in range(window - 1, n):
        s = 0
        for j in range(i - window + 1, i + 1):
            if is_nan(position[j]):
                continue
            for st in states:
                if position[j] == st:
                    s += 1
                    break
        prob[i] = float(s) / float(window) * 100.0 
    return prob      

def cross(long, short):
    p0, p1 = long
    q0, q1 = short
    
    if q0 == q1 :
        return 0
    if p0 == p1:
        return 0
    if p0 >= q0 and p1 <= q1:
        return 1
    elif p0 <= q0 and p1 >= q1:
        return -1
    return 0       


def wakeup(long, short, range_signal, is_up):
    n = len(long)
    direction = 1 if is_up else -1
    x = []
    for i in range(1, n):
        if cross(long[ i - 1: i +1], short[i - 1: i +1]) == direction:
            x.append(i)
    x.append(n - 1)
    trend = full(n, 0)   
    for i in range(len(x) - 1):
        x0 = x[i]
        x1 = x[i + 1]
        for j in range(x0 + 3, x1):
            before = j - 3
            l0 = long[x0]
            s0 = short[x0]
            l1 = long[before]
            s1 = short[before]
            d1 = s1 - s0
            l2 = long[j]            
            s2 = short[j]
            d2 = s2 - s1
            w2 = range_signal[j]
            if is_up:
                if d1 > 0 and d2 > 0 and (s2 - l2) > w2:
                    trend[j] = 1
            else:
                if d1 < 0 and d2 < 0 and (l2 - s2) < -w2:
                    trend[j] = -1 
    return trend, x

def wakeup2(long, short, range_signal, is_up):
    n = len(long)
    trend = full(n, 0)   
    for j in range(5, n):
        before = j - 5
        l1 = long[before]
        s1 = short[before]
        l2 = long[j]            
        s2 = short[j]
        d2 = s2 - s1
        w2 = range_signal[j]
        if is_up:
            if d2 > 0 and (s2 - l2) > w2:
                trend[j] = 1
        else:
            if d2 < 0 and (l2 - s2) < -w2:
                trend[j] = -1 
    return trend

def wakeup3(long, mid, short, width, is_up, is_slow=True):
    n = len(long)
    direction = 1 if is_up else -1
    x = []
    for i in range(1, n):
        if is_slow:
            if cross(long[ i - 1: i +1], mid[i - 1: i +1]) == direction:
                x.append(i)
        else:
            if cross(mid[ i - 1: i +1], short[i - 1: i +1]) == direction:
                x.append(i)
            
    x.append(n - 1)
    trend = full(n, 0)   
    for i in range(len(x) - 1):
        x0 = x[i]
        x1 = x[i + 1]
        for j in range(x0 + 5, x1):
            half = int((j - x0 ) / 2 + x0)
            l0 = long[half]
            m0 = mid[half]
            s0 = short[half]
            l1 = long[j]            
            m1 = mid[j]
            s1 = short[j]
            w1 = width[j]
            if is_up:
                if l1 > l0 and m1 > m0 and s1 > s0 and (s1 - m1) > w1 and (m1 - l1) > w1:
                    trend[j] = 1
            else:
                if l1 < l0 and m1 < m0 and s1 < s0 and (m1 - s1) > w1 and (l1 - m1) > w1:
                    trend[j] = -1 
    return trend, x
    
def ascend(vector, range_signal, count=3):
    n = len(vector)
    asc = full(n, 0)
    for i in range(count, n):
        if vector[i] > range_signal[i] and vector[i - count] < vector[i]:
            asc[i] = 1
        if vector[i] < - range_signal[i] and vector[i - count] > vector[i]:
            asc[i] = -1
    return asc
        
def adx_filter(signal, adx, threshold):
    n = len(signal)
    trend = full(n, 0)
    for i in range(3, n):
        if adx[i] > threshold and adx[i] > adx[i - 3]:
            if signal[i] == 1:
                trend[i] = 1
            elif signal[i] == -1:
                trend[i] = -1
    return trend    

def MA(dic: dict, long_term, short_term):
    ma_long = sma(dic[Columns.CLOSE], long_term)
    dic[Indicators.MA_LONG] = ma_long
    ma_short = sma(dic[Columns.CLOSE], short_term)
    dic[Indicators.MA_SHORT] = ma_short
        
def MABAND( dic: dict, short: int, mid: int, long: int, di_window: int, adx_window: int, adx_threshold: float):
    cl = dic[Columns.CLOSE]
    hi = dic[Columns.HIGH]
    lo  = dic[Columns.LOW]
    
    ema_short = ema(cl, short)
    sma_mid = sma(cl, mid)
    sma_long = sma(cl, long)
    band = ema_short - sma_mid
    adx, _, _ = ADX(hi, lo, cl, di_window, adx_window)
    atr = calc_atr(dic, 4 * 24)

    dic[Indicators.MA_SHORT] = ema_short
    dic[Indicators.MA_MID] = sma_mid
    dic[Indicators.MA_LONG] = sma_long
    dic[Indicators.ADX] = adx
    dic[Indicators.ATR] = atr
    dic[Indicators.MABAND] = band

def MABAND_SIGNAL(dic: dict):
    atr = dic[Indicators.ATR]
    band = dic[Indicators.MABAND]
    ma = dic[Indicators.MA_MID]
    threshold = math.sqrt(atr[-1]) * 5
    print('ATR', atr[-1], threshold)
    
    up, down, signal = detect_cross(band, atr / 2)   
    print(up)
    print(down)
    n = len(band)
    delay = 1
    rate = 0.8
    long = full(n, 0)
    
    """
    for x0, x1 in up:
        long[x0 + delay] = 1
        for x in range(x0 + 1, x1):
            d = ma[x0 + 1: x + 1]
            if max(d) == ma[x]:
                long[x] = -1
            else:
                if (ma[x + 1] - ma[x0]) / (max(d) - ma[x0]) < rate:S
                    long[x] = -1                
    short = full(n, 0)
    for x0, x1 in down:
        short[x0 + delay] = 1
        for x in range(x0 + 1, x1):
            d = ma[x0 + 1: x + 1]
            if min(d) == ma[x0]:
                short[x] = -1 
            else:
                if (ma[x + 1] - ma[x0]) / (min(d) - ma[x0]) < rate:
                    short[x] = -1   
    dic[Indicators.MABAND_LONG] = long
    dic[Indicators.MABAND_SHORT] = short
    """
    return up, down


def detect_cross(band, range_signal):
    n = len(band)
    sig = full(n, 0)
    for i in range(1, n):
        threshold = range_signal[i]
        p0 = band[i - 1]
        p1 = band[i]
        if p0 <= threshold and p1 > threshold:
            sig[i] = 1
        elif p0 >= -threshold and p1 < -threshold:
            sig[i] = -1
            
    up = []
    down = []
    signal = full(n, 0)
    state = 0
    begin = None
    for i, s in enumerate(sig):
        if state == 0:
            state = s
            signal[i] = s
            begin = i
        elif state == 1:
            if s == -1:
                up.append([begin, i - 1])
                begin = i
                state = s
        elif state == -1:
            if s == 1:
                down.append([begin, i - 1])
                begin = i
                state = s
    return up, down, signal

    
def EMABREAK( dic: dict, short: int, long: int, di_window: int, adx_window: int, adx_threshold: float):
    cl = dic[Columns.CLOSE]
    hi = dic[Columns.HIGH]
    lo  = dic[Columns.LOW]
    
    ema_short = ema(cl, short)
    sma_long = sma(cl, long)

    
    dic[Indicators.MA_SHORT] = ema_short
    dic[Indicators.MA_LONG] = sma_long
    adx, _, _ = ADX(hi, lo, cl, di_window, adx_window)
    dic['ADX'] = adx
    
    n = len(cl)
    trend = full(n, 0)
    for i in range(3, n):
        if adx[i] > adx_threshold and adx[i] > adx[i - 3]:
            if ema_short[i] > sma_long[i] and lo[i] > ema_short[i]:
                trend[i] = 1 
            if ema_short[i]  < sma_long[i] and lo[i] < ema_short[i]:
                trend[i] = -1 
    
    up, down, up_event, down_event = detect_signal(trend)
    dic[Indicators.EMABREAK] = trend
    dic[Indicators.EMABREAK_LONG] = up
    dic[Indicators.EMABREAK_SHORT] = down
    

def detect_signal(data):
    up_event = []
    down_event = []
    n = len(data)    
    up = full(n, 0)
    down = full(n, 0)
    active = 0
    for i in range(n):
        if active == 1:
            if data[i] <= 0:
                up_event.append([begin, i])
                up[i] = -1
                active = 0
        elif active == -1:
            if data[i] >= 0:
                down[i] = -1
                down_event.append([begin, i])
                active = 0
        else:
            if data[i] == 1:
                up[i] = 1
                begin = i
                active = 1
            elif data[i] == -1:
                down[i] = 1
                begin = i
                active = -1
    return up, down, up_event, down_event
                
                
def calc_atr(dic, window, how='sma'):
    hi = dic[Columns.HIGH]
    lo = dic[Columns.LOW]
    cl = dic[Columns.CLOSE]
    tr = true_range(hi, lo, cl)
    if how == 'sma':
        atr = sma(tr, window)
    elif how == 'ema':
        atr = ema(tr, window)
    return atr

def ATR(dic: dict, term: int, term_long:int):
    hi = dic[Columns.HIGH]
    lo = dic[Columns.LOW]
    cl = dic[Columns.CLOSE]
    term = int(term)
    tr = true_range(hi, lo, cl)
    dic[Indicators.TR] = tr
    atr = sma(tr, term)
    dic[Indicators.ATR] = atr
    if term_long is not None:
        atr_long = sma(tr, term_long)
        dic[Indicators.ATR_LONG] = atr_long
        
        
        
def ATRP(dic: dict, window, ma_window=0):
    hi = dic[Columns.HIGH]
    lo = dic[Columns.LOW]
    cl = dic[Columns.CLOSE]
    window = int(window)
    tr = true_range(hi, lo, cl)
    dic[Indicators.TR] = tr
    atr = sma(tr, window)
    dic[Indicators.ATR] = atr

    n = len(cl)
    atrp = nans(n)
    for i in range(n):
        a = atr[i]
        c = cl[i]
        if is_nans([a, c]):
            continue
        atrp[i] = a / c * 100.0 
        
    if ma_window > 0:
        atrp = sma(atrp, ma_window)        
    dic[Indicators.ATRP] = atrp


def ADX(hi, lo, cl, di_window: int, adx_term: int):
    tr = true_range(hi, lo, cl)
    n = len(hi)
    dmp = nans(n)     
    dmm = nans(n)     
    for i in range(1, n):
        p = hi[i]- hi[i - 1]
        m = lo[i - 1] - lo[i]
        dp = dn = 0
        if p >= 0 or n >= 0:
            if p > m:
                dp = p
            if p < m:
                dn = m
        dmp[i] = dp
        dmm[i] = dn
    dip = nans(n)
    dim = nans(n)
    dx = nans(n)
    for i in range(di_window - 1, n):
        s_tr = sum(tr[i - di_window + 1: i + 1])
        s_dmp = sum(dmp[i - di_window + 1: i + 1])
        s_dmm = sum(dmm[i - di_window + 1: i + 1])
        dip[i] = s_dmp / s_tr * 100 
        dim[i] = s_dmm / s_tr * 100
        if (dip[i] + dim[i]) == 0:
            dx[i] = 0.0
        else:
            dx[i] = abs(dip[i] - dim[i]) / (dip[i] + dim[i]) * 100
            if dx[i] < 0:
                dx[i] = 0.0
    adx = sma(dx, adx_term)
    return adx, dip, dim
        
        
    
def POLARITY(data: dict, window: int):
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    tr = data[Indicators.TR]
    n = len(hi)
    dmp = nans(n)     
    dmm = nans(n)     
    for i in range(1, n):
        p = hi[i]- hi[i - 1]
        m = lo[i - 1] - lo[i]
        dp = dn = 0
        if p >= 0 or n >= 0:
            if p > m:
                dp = p
            if p < m:
                dn = m
        dmp[i] = dp
        dmm[i] = dn
    dip = nans(n)
    dim = nans(n)
    for i in range(window - 1, n):
        s_tr = sum(tr[i - window + 1: i + 1])
        s_dmp = sum(dmp[i - window + 1: i + 1])
        s_dmm = sum(dmm[i - window + 1: i + 1])
        dip[i] = s_dmp / s_tr * 100 
        dim[i] = s_dmm / s_tr * 100
    
    di = subtract(dip, dim)
    pol = nans(n)
    for i in range(n):
        if is_nan(di[i]):
            continue
        if di[i] > 0:
            pol[i] = UP
        elif di[i] < 0:
            pol[i] = DOWN
    data[Indicators.POLARITY] = pol  
    
def BBRATE(data: dict, window: int, ma_window):
    cl = data[Columns.CLOSE]
    n = len(cl)
    std = nans(n)     
    for i in range(window - 1, n):
        d = cl[i - window + 1: i + 1]    
        std[i] = np.std(d)   
    ma = sma(cl, ma_window)     
    rate = nans(n)
    for i in range(n):
        c = cl[i]
        m = ma[i]
        s = std[i]
        if is_nans([c, m, s]):
            continue
        rate[i] = (cl[i] - ma[i]) / s * 100.0
    data[Indicators.BBRATE] = rate

def BB(data: dict, window: int, ma_window:int, band_multiply):
    cl = data[Columns.CLOSE]
    n = len(cl)
    #ro = roi(cl)
    std = nans(n)     
    for i in range(window - 1, n):
        d = cl[i - window + 1: i + 1]    
        std[i] = np.std(d)   
    ma = sma(cl, ma_window)     
        
    upper, lower = band(ma, std, band_multiply)    
    data[Indicators.BB] = std
    data[Indicators.BB_UPPER] = upper
    data[Indicators.BB_LOWER] = lower
    data[Indicators.BB_MA] = ma
    
    pos = band_position(cl, lower, ma, upper)
    up = probability(pos, [1, 2], 50)
    down = probability(pos, [-1, -2], 50)
    data[Indicators.BB_UP] = up
    data[Indicators.BB_DOWN] = down
    
    cross_up, cross_down, cross = cross_value(up, 50)
    data[Indicators.BB_CROSS] = cross
    data[Indicators.BB_CROSS_UP] = cross_up
    data[Indicators.BB_CROSS_DOWN] = cross_down

def time_jst(year, month, day, hour=0):
    t0 = datetime(year, month, day, hour)
    t = t0.replace(tzinfo=JST)
    return t

def pivot2(signal, threshold, left_length=2, right_length=2):
    n = len(signal)
    out = full(n, np.nan) 
    out_mid = full(n, np.nan)
    for i in range(left_length + right_length, n):
        if is_nans(signal[i - right_length - right_length: i + 1]):
            continue
        center = signal[i - right_length]
        left = signal[i - left_length - right_length: i - right_length]
        range_left = abs(max(left) - min(left))
        right = signal[i - right_length + 1: i + 1]
        d_right = np.mean(right) - center
        
        if range_left < 5:
            if center >= 90 and d_right < -threshold:
                if np.nanmin(out[i - 10: i]) != Signal.SHORT:
                    out[i] = Signal.SHORT
            elif center <= 10 and d_right > threshold:
                if np.nanmax(out[i - 10: i]) != Signal.LONG:
                    out[i] = Signal.LONG
                                
            if center >= 40 and center <= 60:
                if d_right < -threshold:
                    if np.nanmin(out_mid[i - 10: i]) != Signal.SHORT:
                        out_mid[i] = Signal.SHORT 
                elif d_right > threshold:
                    if np.nanmax(out_mid[i - 10: i]) != Signal.LONG:
                        out_mid[i] = Signal.LONG 
    return out, out_mid

def vwap_rate(price, vwap, std, median_window, ma_window):
    n = len(price)
    rate = nans(n)
    i = -1
    for p, v, s in zip(price, vwap, std):
        i += 1
        if is_nans(([p, v, s])):
            continue
        if s != 0.0:
            r = (p - v) / s * 100.0
            rate[i] = r #20 * int(r / 20)        
    med = median(rate, median_window)        
    ma = sma(med, ma_window)
    return ma

def vwap_pivot(signal, threshold, left_length, center_length, right_length):
    n = len(signal)
    out = full(n, np.nan) 
    for i in range(left_length + center_length + right_length, n):
        if is_nans(signal[i - right_length - center_length - right_length: i + 1]):
            continue
        l = i - left_length - center_length - right_length + 1
        c = i - right_length - center_length + 1
        r = i - right_length + 1
        left = signal[l: c]
        center = np.mean(signal[c: r])
        right = signal[r: i + 1]
        
        polarity = 0
        # V peak
        d_left = np.nanmax(left) - center
        d_right = np.nanmax(right) - center
        if d_left > 0 or d_right > 0:
            if d_left >= threshold and d_right >= threshold:
                polarity = 1
        # ^ Peak
        d_left = center - np.nanmin(left)
        d_right = center - np.nanmin(right)
        if d_left > 0 and d_right > 0:
            if d_left >= threshold and d_right >= threshold:
                polarity = -1
        
        if polarity == 0:      
            sig = np.nan
        elif polarity > 0:
            sig = Signal.LONG
        elif polarity < 0:
            sig = Signal.SHORT

        """
        if center >= 200:
            if sig == Signal.LONG:
                sig = np.nan
                    
        if center > -50 and center < 50:
            sig = np.nan
    
        if center <= -200:
            if sig == Signal.SHORT:
                sig = np.nan            
        """
        
        if sig == Signal.SHORT:
            if np.nanmin(out[i - 10: i]) == Signal.SHORT:
                sig = np.nan
        elif sig == Signal.LONG:
            if np.nanmax(out[i - 10: i]) == Signal.LONG:
                sig = np.nan
                           
        out[i] = sig
    return out

def VWAP(data: dict, begin_hour_list, pivot_threshold, pivot_left_len, pivot_center_len, pivot_right_len, median_window, ma_window):
    jst = data[Columns.JST]
    n = len(jst)
    MID(data)
    mid = data[Columns.MID]
    volume = data[Columns.VOLUME]
    
    vwap = full(n, np.nan)
    power_acc = full(n, np.nan)
    volume_acc = full(n, np.nan)
    std = full(n, 0)
    valid = False
    for i in range(n):
        t = jst[i]
        if t.hour in begin_hour_list:
            if t.minute == 0 and t.second == 0:
                power_sum = 0
                vwap_sum = 0
                volume_sum = 0
                valid = True
        if valid:
            vwap_sum += volume[i] * mid[i]
            volume_sum += volume[i]  
            volume_acc[i] = volume_sum
            power_sum += volume[i] * mid[i] * mid[i]  
            if volume_sum > 0:
                vwap[i] = vwap_sum / volume_sum
                power_acc[i] = power_sum
                deviation = power_sum / volume_sum - vwap[i] * vwap[i]
                if deviation > 0:
                    std[i] = np.sqrt(deviation)
                else:
                    std[i] = 0
    data[Indicators.VWAP] = vwap
    rate = vwap_rate(mid, vwap, std, median_window, ma_window)
    data[Indicators.VWAP_RATE] = rate
    
    dt = jst[1] - jst[0]
    data[Indicators.VWAP_SLOPE] = slope(vwap, 10, dt.total_seconds() / 60)
    
    for i in range(1, 5):
        upper, lower = band(vwap, std, float(i))
        data[Indicators.VWAP_UPPER + str(i)] = upper
        data[Indicators.VWAP_LOWER + str(i)] = lower
    
    signal1 = vwap_pivot(rate, pivot_threshold, pivot_left_len, pivot_center_len, pivot_right_len)
    data[Indicators.VWAP_RATE_SIGNAL] = signal1    
    pos = band_position(mid, lower, vwap, upper)
    up = probability(pos, [1, 2], 40)
    down = probability(pos, [-1, -2], 40)
    data[Indicators.VWAP_PROB] = up
    data[Indicators.VWAP_DOWN] = down
    
    signal2 = slice_vector(up, 90, 10, 10)
    data[Indicators.VWAP_PROB_SIGNAL] = signal2
    
    
def slice_vector(vector, threshold_upper: float, threshold_lower: float, length: int):
    n = len(vector)
    states = nans(n)
    begin = None
    state = 0
    for i in range(n):
        if state == 0:
            if vector[i] >= threshold_upper:
                state = 1
                begin = i
            elif vector[i] <= threshold_lower:
                state = -1
                begin = i
        elif state == 1:
            if vector[i] < threshold_upper:
                state = 0
                if (i - begin + 1) >= length:
                    states[i] = Signal.SHORT
        elif state == -1:
            if vector[i] > threshold_lower:
                state = 0
                if (i - begin + 1) >= length:
                    states[i] = Signal.LONG
    return states            
    
def RCI(data: dict, window: int, pivot_threshold: float, pivot_length: int):
    cl = data[Columns.CLOSE]
    rc = rci(cl, window)
    data[Indicators.RCI] = rc
    signal = slice_vector(rc, pivot_threshold, -pivot_threshold, pivot_length)
    data[Indicators.RCI_SIGNAL] = signal
    
       
def band(vector, signal, multiply):
    n = len(vector)
    upper = nans(n)
    lower = nans(n)
    for i in range(n):
        upper[i] = vector[i] + multiply * signal[i]
        lower[i] = vector[i] - multiply * signal[i]
    return upper, lower



def volatility(data: dict, window: int):
    time = data[Columns.TIME]
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(cl)
    volatile = nans(n)
    for i in range(window, n):
        d = []
        for j in range(i - window + 1, i + 1):
            d.append(cl[j - 1] - op[j])
            if cl[j] > op[j]:
                # positive
                d.append(lo[j] - op[j])
                d.append(hi[j] - lo[j])
                d.append(cl[j] - hi[j])
            else:
                d.append(hi[j] - op[j])
                d.append(lo[j] - hi[j])
                d.append(cl[j] - lo[j])
        sd = stat.stdev(d)
        volatile[i] = sd / float(window) / op[i] * 100.0
    return               
            
def TREND_ADX_DI(data: dict, adx_threshold: float):
    adx = data[Indicators.ADX]
    adx_slope = slope(adx, 5)
    di_p = data[Indicators.DI_PLUS]
    di_m = data[Indicators.DI_MINUS]
    n = len(adx)
    trend = full(n, 0)
    for i in range(n):
        if adx[i] > adx_threshold and adx_slope[i] > 0: 
            delta = di_p[i] - di_m[i]
            if delta > 0:
                trend[i] = UP
            elif delta < 0:
                trend[i] = DOWN
    data[Indicators.TREND_ADX_DI] = trend

def MID(data: dict):
    cl = data[Columns.CLOSE]
    op = data[Columns.OPEN]
    n = len(cl)
    md = nans(n)
    for i in range(n):
        o = op[i]
        c = cl[i]
        if is_nans([o, c]):
            continue
        md[i] = (o + c) / 2
    data[Columns.MID] = md
    
def ATR_TRAIL(data: dict, atr_window: int, atr_multiply: float, peak_hold_term: int, horizon: int):
    atr_window = int(atr_window)
    atr_multiply = int(atr_multiply)
    peak_hold_term = int(peak_hold_term)
    time = data[Columns.TIME]
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(cl)
    ATR(data, atr_window, None)
    atr = data[Indicators.ATR]
    stop = nans(n)
    for i in range(n):
        h = hi[i]
        a = atr[i]
        if is_nans([h, a]):
            continue
        stop[i] = h - a * atr_multiply
        
    trail_stop = nans(n)
    for i in range(n):
        d = stop[i - peak_hold_term + 1: i + 1]
        if is_nans(d):
            continue
        trail_stop[i] = max(d)
        
    trend = full(n, np.nan)
    up = full(n, np.nan)
    down = full(n, np.nan)
    for i in range(n):
        c = cl[i]
        s = trail_stop[i]
        if is_nans([c, s]):
            continue
        if c > s:
            trend[i] = UP
            up[i] = s
        else:
            trend[i] = DOWN
            down[i] = s
            
    data[Indicators.ATR_TRAIL_UP] = up
    data[Indicators.ATR_TRAIL_DOWN] = down
            
    break_signal = full(n, np.nan)
    for  i in range(1, n):
        if trend[i - 1] == UP and trend[i] == DOWN:
            break_signal[i] = DOWN
        if trend[i - 1] == DOWN and trend[i] == UP:
            break_signal[i] = UP

    signal = full(n, np.nan)
    for i in range(horizon, n):
        brk = break_signal[i - horizon]
        if brk == DOWN and trail_stop[i] > cl[i]:
            signal[i] = Signal.SHORT
        elif brk == UP and trail_stop[i] < cl[i]:
            signal[i] = Signal.LONG        
            
    data[Indicators.ATR_TRAIL] = trail_stop
    data[Indicators.ATR_TRAIL_SIGNAL] = signal

             
def SUPERTREND(data: dict,  atr_window: int, multiply, column=Columns.MID):
    time = data[Columns.TIME]
    if column == Columns.MID:
        MID(data)
    price = data[column]
    n = len(time)
    atr = calc_atr(data, atr_window)
    ATRP(data, atr_window, atr_window)
    atr_u, atr_l = band(data[column], atr, multiply)
    data[Indicators.ATR_UPPER] = atr_u
    data[Indicators.ATR_LOWER] = atr_l

    
def SUPERTREND_SIGNAL(data: dict, short_term):
    time = data[Columns.TIME]
    n = len(time)
    cl = data[Columns.CLOSE]
    atr_u = data[Indicators.ATR_UPPER]
    atr_l = data[Indicators.ATR_LOWER]
    price = sma(cl, short_term)
    
    trend = nans(n)
    sig = full(n, 0)
    stop_price = nans(n)
    upper = nans(n)
    lower = nans(n)
    is_valid = False
    for i in range(1, n):
        if is_valid == False:
            if is_nans([atr_l[i - 1], atr_u[i - 1]]):
                continue
            else:
                lower[i - 1] = atr_l[i - 1]
                trend[i - 1] = UP
                is_valid = True            
        if trend[i - 1] == UP:
            # up trend
            if np.isnan(lower[i - 1]):
                lower[i] = atr_l[i -1]
            else:
                if atr_l[i] > lower[i - 1]:
                    lower[i] = atr_l[i]
                else:
                    lower[i] = lower[i - 1]
            if price[i] < lower[i]:
                 # up->down trend 
                trend[i] = DOWN
                sig[i] = Signal.SHORT
                stop_price[i] = lower[i]
            else:
                trend[i] = UP
        else:
            # down trend
            if np.isnan(upper[i - 1]):
                upper[i] = atr_u[i]
            else:
                if atr_u[i] < upper[i - 1]:
                    upper[i] = atr_u[i]
                else:
                    upper[i] = upper[i - 1]
                    
            if price[i] > upper[i]:
                # donw -> up trend
                trend[i] = UP
                sig[i] = Signal.LONG
                stop_price[i] = upper[i]
            else:
                trend[i] = DOWN
    
    exits = full(n, 0)
    ent = False
    for i in range(n):
        if sig[i] != 0:
            if ent:
                if sig[i] == Signal.LONG:
                    exits[i] = Signal.SHORT
                elif sig[i] == Signal.SHORT:
                    exits[i] = Signal.LONG
            else:
                ent = True
           
    data[Indicators.SUPERTREND_MA] = price
    data[Indicators.SUPERTREND] = trend  
    data[Indicators.SUPERTREND_ENTRY] = sig
    data[Indicators.SUPERTREND_EXIT]  = exits     
    data[Indicators.SUPERTREND_U] = upper  
    data[Indicators.SUPERTREND_L] = lower  
    return 

def MAGAP(timeframe, data: dict, long_term, mid_term, short_term, tap):
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(op)
    ma_long = sma(cl, long_term)
    ma_mid = sma(cl, mid_term)
    ma_short = sma(cl, short_term)
    data[Indicators.MA_LONG] = ma_long
    data[Indicators.MA_MID] = ma_mid
    data[Indicators.MA_SHORT] = ma_short
    
    gap = [0 for _ in range(n)]
    for i in range(n):
        gap[i] = (ma_short[i] - ma_mid[i]) / ma_mid[i] * 100.0
    data[Indicators.MAGAP] = gap    
    slope = slope_by_hour(timeframe, gap)
    data[Indicators.MAGAP_SLOPE] = slope
 
def MAGAP_SIGNAL(timeframe, data, short_slope_threshold, long_slope_threshold, delay_max):
    gap = data[Indicators.MAGAP]
    slope = data[Indicators.MAGAP_SLOPE]
    trnd = trend(timeframe, data, Indicators.MA_LONG, long_slope_threshold)
    up, down, entry, ext = detect_gap_cross(gap, slope, trnd, short_slope_threshold, delay_max=delay_max)
    data[Indicators.MAGAP_ENTRY] = entry
    data[Indicators.MAGAP_EXIT] = ext
    return up, down

def calc_std(data, window):
    cl = data[Columns.CLOSE]
    n = len(cl)
    std = nans(n)     
    for i in range(window - 1, n):
        d = cl[i - window + 1: i + 1]    
        std[i] = np.std(d) 
    return np.array(std)  


def slice_upper(vector, value):
    n = len(vector)
    out = np.full(n, np.nan)
    for i in range(n):
        if vector[i] > value:
            out[i] = 1
    return out
        
def SQUEEZER(data, window, multiply, length, k_atr=2.0):
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(cl)
    std = calc_std(data, window) * multiply
    atr = calc_atr(data, window, how='ema')
    atr2 = calc_atr(data, window)
    dif = atr - std
    sqz = slice_upper(dif, 0)
    terms = detect_terms(sqz, 1)
    points = []
    for i in range(len(terms)):
        if i == 0:
            points.append([terms[i][0], 0])
            continue
        begin0, _ = terms[i - 1]
        begin1, _ = terms[i]
        if cl[begin1] > cl[begin0]:
            points.append([begin1, 1])
        elif cl[begin1] < cl[begin0]:
            points.append([begin1, -1])
        else:
            points.append([begin1, 0])

    upper = np.full(n, np.nan)
    upper_sl = np.full(n, np.nan)
    lower = np.full(n, np.nan)   
    lower_sl = np.full(n, np.nan)
    signal = np.full(n, 0)         
    for i in range(2, len(points)):
        begin0, _ = points[i - 2] 
        begin1, s = points[i - 1]
        begin2, _ = points[i]
        if s == 1:
            signal[begin1] = 1
        elif s == -1:
            signal[begin1] = -1
        if i == len(points) - 1:
            iend = n - 1
        else:
            iend = (begin1 + length) % begin2
        for j in range(begin1, iend):
            if s == 1:
                upper[j] = cl[begin1] + atr2[j] * k_atr
                lower[j] = cl[begin0] - atr2[j] * k_atr
            elif s == -1:
                lower[j] = cl[begin1] - atr2[j] * k_atr
                upper[j] = cl[begin0] + atr2[j] * k_atr
                
        if i == len(points) - 1:
            iend = n - 1
        else:
            iend = begin2                
        for j in range(begin1, iend):
            if s == 1:
                upper_sl[j] = cl[begin1] - atr2[j] * k_atr
                lower_sl[j] = cl[begin0] + atr2[j] * k_atr
            elif s == -1:
                lower_sl[j] = cl[begin1] + atr2[j] * k_atr
                upper_sl[j] = cl[begin0] - atr2[j] * k_atr        
            
            
    entry = np.full(n, 0)
    for i in range(1, n):
        if cl[i - 1] <= upper[i - 1] and cl[i] > upper[i]:
            entry[i] = 1
        if cl[i - 1] >= lower[i - 1] and cl[i] < lower[i]:
            entry[i] = -1        
    
    ext = np.full(n, 0)
    for i in range(1, len(points)):
        index0, status0 = points[i - 1]
        index1, status1 = points[i]
        if (status0 * status1) == -1:
            ext[index1] = 1    
        
    data[Indicators.SQUEEZER] = sqz
    data[Indicators.SQUEEZER_STD] = std
    data[Indicators.SQUEEZER_ATR] = atr
    data[Indicators.SQUEEZER_UPPER] = upper
    data[Indicators.SQUEEZER_UPPER_SL] = upper_sl
    data[Indicators.SQUEEZER_LOWER] = lower
    data[Indicators.SQUEEZER_LOWER_SL] = lower_sl
    data[Indicators.SQUEEZER_SIGNAL] = signal
    data[Indicators.SQUEEZER_ENTRY] = entry
    data[Indicators.SQUEEZER_EXIT] = ext
    
def detect_terms(vector, value):
    terms = []
    n = len(vector)
    begin = None
    for i in range(n):
        if begin is None:
            if vector[i] == value:
                begin = i
        else:
            if vector[i] != value:
                terms.append([begin, i - 1])
                begin = None
    if begin is not None:
        terms.append([begin, n - 1])
    return terms

def PPP(timeframe, data: dict, long_term, mid_term, short_term, pre, post, target, threshold=0.01, tap=0):
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(op)
    ma_long = sma(cl, long_term)
    ma_mid = sma(cl, mid_term)
    ma_short = sma(cl, short_term)
    data[Indicators.MA_LONG] = ma_long
    data[Indicators.MA_MID] = ma_mid
    data[Indicators.MA_SHORT] = ma_short    
    slope_long = slope_by_hour(timeframe, ma_long, tap=20)
    slope_mid = slope_by_hour(timeframe, ma_mid, tap=20)
    slope_short = slope_by_hour(timeframe, ma_short, tap=20)
    data[Indicators.MA_LONG_SLOPE] = slope_long
    data[Indicators.MA_MID_SLOPE] = slope_mid
    data[Indicators.MA_SHORT_SLOPE] = slope_short
    ATRP(data, 40, ma_window=40)
    cross = full(n, 0)
    for i in range(n):
        if ma_short[i] > ma_mid[i] and ma_mid[i] > ma_long[i]:
            if slope_long[i] > threshold and slope_mid[i] > threshold and slope_short[i] > threshold:
                cross[i] = 1
        elif ma_short[i] < ma_mid[i] and ma_mid[i] < ma_long[i]:
            if slope_long[i] < -threshold and slope_mid[i] < -threshold and slope_short[i] < -threshold:
                cross[i] = - 1
                
    
    up, down = majority(cross)
    entries = full(n, 0)
    for i in range(1, n):
        if up[i - 1] == 0 and up[i] == 1:
            entries[i] = Signal.LONG
        elif down[i - 1] == 0 and down[i] == 1:
            entries[i] = Signal.SHORT
    
    polarity = full(n, 0)    
    exits = full(n, 0)   
    current = 0
    for i in range(n):
        if current == 0:
            if up[i] != 0:
                current = 1
            elif down[i] != 0:
                current = -1
        elif current == 1:
            if down[i] == 1:
                current = -1
                exits[i] = 1
        elif current == -1:
            if up[i] == 1:
                current = 1
                exits[i] = -1
        polarity[i] = current
          
    data[Indicators.MA_GOLDEN_CROSS] = cross
    data[Indicators.PPP_ENTRY] = entries
    data[Indicators.PPP_EXIT] = exits
    data[Indicators.PPP_UP] = up
    data[Indicators.PPP_DOWN] = down
    

def majority(vector, values=[1, -1], window=10, rate=0.8):
    n = len(vector)
    results = []
    for value in values:
        data = [1 if v == value else 0 for v in vector]
        sig = full(n, 0)
        for i in range(window - 1, n):
            d = data[i - window + 1: i + 1]
            if (sum(d) / window) >= rate:
                sig[i] = 1
        results.append(sig)
    return results
    
    
    
    
def seek(vector, begin, value):
    n = len(vector)
    for i in range(begin, n):
        if vector[i] == value:
            return i
    return n + 1


def slope_by_hour(timeframe, vector, tap=10):
    n = len(vector)
    if timeframe[0] == 'M':
        hour = int(timeframe[1:]) / 60  * tap
    elif timeframe[0] == 'H':
        hour = int(timeframe[1:]) * tap
    else:
        raise Exception('error')
        
    slope = full(n, 0.0)
    for i in range(tap, n):
        if not np.isnan(vector[i - tap + 1]):
            slope[i] = (vector[i] - vector[i - tap + 1]) / hour / vector[i - tap + 1] * 100.0
    return slope

def trend(timeframe, data, column, slope_threshold):
    ma = data[column]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    slope = slope_by_hour(timeframe, ma, tap=20)
    data['MA_LONG_SLOPE'] = slope
    
    n = len(ma)
    out = full(n, 0)
    for i, (h, l, m) in enumerate(zip(hi, lo, ma)):
        if abs(slope[i]) < slope_threshold:
            continue
        if l > m:
            out[i] = 1 
        elif h < m:
            out[i] = -1
    return out 

def detect_gap_breakout(gap, term1, term2, threshold, level):
    term = term1 + term2
    n = len(gap)
    signal = full(n, 0)
    for i in range(term - 1, n):
        d = gap[i - term + 1: i - term + term1]
        height = max(d) - min(d)
        if height < threshold:
            if gap[i] >= max(d) + threshold:
                signal[i] = Signal.LONG
            elif gap[i] <=  min(d) - threshold:
                signal[i] = Signal.SHORT
    
    current = 0 
    up = []
    down = []
    order_signal = full(n, 0)
    for i in range(n):
        if signal[i] == 0:
            continue
        if current != signal[i]:
            current = signal[i]
            if signal[i] == Signal.LONG:
                if gap[i] > level or gap[i] < -level:
                    up.append(i)
                    order_signal[i] = Signal.LONG
            elif signal[i] == Signal.SHORT:
                if gap[i] > level or gap[i] < -level:
                    down.append(i)
                    order_signal[i] = Signal.SHORT
    return up, down, signal

    

    
def detect_gap_cross(gap, slope, trend, threshold, delay_max=12):
    n = len(gap)
    sig_xup = full(n, 0)
    sig_xdown = full(n, 0)
    count = 0
    for i in range(1, n):
        if gap[i - 1] < 0 and gap[i] >= 0:
            sig_xup[i] = 1
            count += 1
        elif gap[i - 1]  > 0 and gap[i] <= 0:
            sig_xdown[i] = 1
            count += 1
             
    #print('#1', count)

    count = 0             
    sig_up = full(n, 0)
    for i in range(delay_max, n):
        d = sig_xup[i - delay_max: i + 1]
        if max(d) == 1 and slope[i] > threshold:
            if trend[i] == 1:
                sig_up[i] = 1
                count += 1
                
    #print('#2', count)
    sig_down = full(n, 0)
    for i in range(delay_max, n):
        d = sig_xdown[i - delay_max: i + 1]
        if max(d) == 1 and slope[i] < -threshold:
            if trend[i] == -1:
                sig_down[i] = 1
                count += 1
                
    #print('#3', count)
                      
    current = None
    entry = full(n, 0)
    
    count = 0
    xup = []
    xdown = []
    for i in range(n):
        if current is None:
            if sig_up[i] == 1:
                entry[i] = Signal.LONG
                current = Signal.LONG
                xup.append(i)
                count += 1
            elif sig_down[i] == 1:
                entry[i] = Signal.SHORT
                current = Signal.SHORT
                xdown.append(i)
                count += 1
        else:
            if current == Signal.LONG:
                if sig_up[i] == 0:
                    current = None
                if sig_down[i] == 1:
                    entry[i] = Signal.SHORT
                    current = Signal.SHORT
                    xdown.append(i)
                    count += 1
            elif current == Signal.SHORT:
                if sig_down[i] == 0:
                    current = None
                if sig_up[i] == 1:
                    entry[i] = Signal.LONG
                    current = Signal.LONG
                    xup.append(i)
                    count += 1

    ext = full(n, 0)
    current = None
    for i in range(n):
        if entry[i] == 0:
            continue
        if current is None:
            current = entry[i]
        else:
            if entry[i] != current:
                ext[i] = 1
                current = entry[i]
    #print('#4', count)
    return xup, xdown, entry, ext
    
def detect_trend_term(vector):
    long = []
    n = len(vector)
    begin = None
    for i in range(n):
        if begin is None:
            if vector[i] > 0:
                begin = i
        else:
            if vector[i] <= 0:
                long.append([begin, i - 1])
                begin = None         
    short = []
    begin = None
    for i in range(n):
        if begin is None:
            if vector[i] < 0:
                begin = i
        else:
            if vector[i] >= 0:
                short.append([begin, i - 1])
                begin = None
    return long, short

def breakout(data: dict, n_bo: int, window: int, term_max):
    cl = data[Columns.CLOSE]
    n = len(cl)
    bo = np.full(n, 0)
    flag = np.full(n, 0)
    prices = np.full(n_bo, 0)
    sl = np.full(n, np.nan)
    i = window - 1
    while i < n:
        d = cl[i - window + 1: i]
        threshold = 0.5 *  (max(d) - min(d))
        if cl[i] > max(d) + threshold:
            prices[0] = cl[i]
            count = 1
            flag[i] = 1
            bo[i] = 1
            for j in range(i + 1, (i + term_max + 1) % n):
                if cl[j] > prices[count - 1]:
                    flag[j] = count + 1
                    sl[j] = prices[0]
                    prices[count] = cl[j]
                    count += 1
                    if count >= n_bo:
                        break
            i += term_max + 1
        elif cl[i]  < min(d) - threshold:
            prices[0] = cl[i]
            count = 1
            flag[i] = -1
            bo[i] = -1
            for j in range(i + 1, (i + term_max + 1) % n):
                if cl[j] < prices[count - 1]:
                    flag[j] = -1 * (count + 1)
                    sl[j] = prices[0]
                    prices[count] = cl[j]
                    count += 1
                    if count >= n_bo:
                        break
            i += term_max + 1
        else:
            i += 1
    data[Indicators.BREAKOUT] = flag
    data[Indicators.BREAKOUT_SL] = sl
    ent = np.full(n, 0)
    for i in range(n):
        if flag[i] == n_bo:
            ent[i] = 1
        elif flag[i] == -1 * n_bo:
            ent[i] = -1
    data[Indicators.BREAKOUT_ENTRY] = ent
    ext = np.full(n, np.nan)
    current = None
    for i in range(n):
        if ent[i] == 0:
            continue
        if current is None:
            current = ent[i]
        else:
            if ent[i] != current:
                ext[i] = 1
                current = ent[i]    
    data[Indicators.BREAKOUT_EXIT] = ext
    
    
    
def rally(data, short_term=12, mid_term=24, long_term=48, threshold=0.1, rate=0.7):
    cl = data[Columns.CLOSE]
    n = len(cl)
    ma1 = sma(cl, short_term)
    ma2 = sma(cl, mid_term)
    ma3 = sma(cl, long_term)
    d1 = diff(ma1)
    d2 = diff(ma2)
    d3 = diff(ma3)
    w1 = np.array(ma1) - np.array(ma2)
    w2 = np.array(ma2) - np.array(ma3)
    r = np.abs(w1 / w2)
    
    c1 = (w1 > 0) and (w2 > 0)
    c2 = (d1 > threshold) and (d2 > threshold) and (d3 > threshold)
    c3 = (r > (1.0 - rate)) and (r < (1.0 + rate))
    
    sig = np.full(n, 0)
    for i in range(n):
        if c1[i] and c2[i] and c3[i]:
            sig[i] = 1
            
    c1 = (w1 < 0) and (w2 < 0)
    c2 = (d1 < threshold) and (d2 < threshold) and (d3 < threshold)
    for i in range(n):
        if c1[i] and c2[i] and c3[i]:
            sig[i] = -1   
            
    data[Indicators.MA_SHORT] = ma1
    data[Indicators.MA_MID] = ma2
    data[Indicators.MA_LONG] = ma3
    data[Indicators.RALLY] = sig 
    
    
def diff(array):
    n = len(array)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = array[i] - array[i - 1]
    return out

def detect_peaks(vector):
   peaks, _ = find_peaks(vector, plateau_size=1)
   return peaks

def test():
    sig = [29301.79, 29332.16, 28487.87, 28478.56, 28222.48,
           28765.66, 28489.13, 28124.28, 28333.52]
    ma = full(-1, len(sig))
    
    x = rci(sig, 9)
    print(x)
    
if __name__ == '__main__':
    test()
    

