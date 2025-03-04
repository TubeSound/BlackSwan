import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.plotting import output_notebook
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import Spacer, Span, Text
from bokeh.models import HoverTool, ColumnDataSource
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

output_notebook()

TOOLS = "pan,wheel_zoom,box_zoom,box_select,crosshair,reset,save"
TOOLTIPS=[  ( 'date',   '@date' ),
            ( 'close',  '@close{0,0}' ), 
        ]

class TimeChart():
    def __init__(self, title, width, height, date, date_format='%Y/%m/%d %H:%M'):
        self.fig = figure(  x_axis_type="linear",
                            tools=TOOLS, 
                            plot_width=width,
                            plot_height=height,
                            tooltips=TOOLTIPS,
                            title = title)
    
        self.date = date
        self.fig.xaxis.major_label_overrides = {i: d.strftime(date_format) for i, d in enumerate(date)}
        self.indices = range(len(date))

    def time_index(self, time):
        for i, d in enumerate(self.date):
            if d > time:
                return i
        return -1      

    def line(self, y, **kwargs):
        self.fig.line(self.indices, np.array(y), **kwargs)
        
    def scatter(self, ts, ys, **kwargs):
        indices = [self.time_index(t) for t in ts]
        self.fig.scatter(indices, np.array(ys), **kwargs)
    
        
    def markers(self, signal, values, status, marker='o', color='black', size=10):
        marks = {'o': 'circle', 'v': 'inversed_triangle', '^': 'triangle', '+': 'cross', 'x': 'x'}
        indices = []
        ys = []
        for i, (s, v) in enumerate(zip(signal, values)):
            if s == status:
                indices.append(i)
                ys.append(v)
        self.fig.scatter(indices, np.array(ys), marker=marks[marker], color=color, size=size)
        
    def vline(self, time, color, width=1):
        index = self.time_index(time)
        span = Span(location=index,
                    dimension='height',
                    line_color=color,
                    line_width=width)
        self.fig.add_layout(span)
        
    def text(self, time, y, text, color):
        glyph = Text(x="x", y="y", text="text",  text_color=color, text_font_size='9pt')
        source = ColumnDataSource(dict(x=[self.time_index(time)], y=[y], text=[text]))
        self.fig.add_glyph(source, glyph)
        
        
