from bokeh.plotting import figure, output_file, save
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.layouts import HBox, VBox
from bokeh.charts import BoxPlot
from bokeh.models import CheckboxGroup, CustomJS
import glob
import pandas as pd
import numpy as np
import re

def plot(title): 
    with open('12.txt', 'r') as f:
        data = f.read()
    a1 = [int(x) for x in re.findall('(\d*) \d* .*', data)]
    a2 = [int(x) for x in re.findall('\d* (\d*) .*', data)]
    times = [float(x) for x in re.findall('\d* \d* (.*)', data)]
    df = pd.DataFrame({'proc':a1, 'thread':a2, 'time':times})
    df.sort_values(['proc'], ascending=[1], inplace=True)
    xs = ['({},{})'.format(x,y) for x,y in zip(df['proc'], df['thread'])]
    p1 = figure(x_range=xs)
    p1.line(xs, df['time'], line_width=2)
    p1.circle(xs, df['time'], size=4)
    p1.xaxis.axis_label = '#proc'
    p1.yaxis.axis_label = 'time(s)'
    
    with open('all.txt', 'r') as f:
        data = f.read()
    a1 = [int(x) for x in re.findall('(\d*) \d* .*', data)]
    a2 = [int(x) for x in re.findall('\d* (\d*) .*', data)]
    xs = [x*y for x,y in zip(a1,a2)]
    times = [float(x) for x in re.findall('\d* \d* (.*)', data)]
    df = pd.DataFrame({'proc':a1, 'thread':a2, 'xs':xs, 'time':times})
    df.sort_values(['xs', 'proc'], ascending=[1,1], inplace=True)
    xs = ['({},{})'.format(x,y) for x,y in zip(df['proc'], df['thread'])]
    p2 = figure(x_range=xs)
    p2.line(xs, df['time'], line_width=2)
    p2.circle(xs, df['time'], size=4)
    p2.xaxis.axis_label = '#proc'
    p2.yaxis.axis_label = 'time(s)'
    p = VBox(p1, p2)
    output_file('best.html')
    save(p)

plot('1000*1000')
