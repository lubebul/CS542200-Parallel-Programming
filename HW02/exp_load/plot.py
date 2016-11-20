from bokeh.plotting import figure, output_file, save
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.layouts import HBox, VBox
from bokeh.models import CheckboxGroup, CustomJS
import glob
import pandas as pd
import numpy as np
import re

def plot(): 
    p1 = figure(title='#points')
    p2 = figure(title='execution time')
    eps = [[0, 0.15], [0.35, 0.45], [0.65,0.75]]
    colors = [['greenyellow', 'darkgreen'], ['lightblue', 'darkblue'], ['pink', 'red']]
    for c0, ver in zip(range(3), ['MPI', 'OpenMP', 'Hybrid']):
        for c1, tp in zip(range(3), ['static', 'dynamic']):
            with open('MS_{}_{}_load.out'.format(ver, tp), 'r') as f:
                data = f.read()
            ids = [int(x) for x in re.findall('\[(\d*)\] \d* \d*.\d*', data)]
            pts = [int(x) for x in re.findall('\[\d*\] (\d*) \d*.\d*', data)]
            times = [float(x) for x in re.findall('\[\d*\] \d* (\d*.\d*)', data)]
            df = pd.DataFrame({'id':ids, 'point':pts, 'time':times})
            df.sort_values(['id'], ascending=[1], inplace=True)
            for idx, row in df.iterrows():
                p1.line([row['id']+eps[c0][c1], row['id']+eps[c0][c1]], [0, row['point']], legend='{}-{}'.format(ver, tp), line_color=colors[c0][c1], line_width=4)
                p1.circle([row['id']+eps[c0][c1]], [row['point']], color=colors[c0][c1], size=5)
                p2.line([row['id']+eps[c0][c1], row['id']+eps[c0][c1]], [0, row['time']], legend='{}-{}'.format(ver, tp), line_color=colors[c0][c1], line_width=4)
                p2.circle([row['id']+eps[c0][c1]], [row['time']], color=colors[c0][c1], size=5)
    p1.xaxis.axis_label = 'thread-ID / MPI_Task-ID'
    p1.yaxis.axis_label = '#points'
    p2.xaxis.axis_label = 'thread-ID / MPI_Task-ID'
    p2.yaxis.axis_label = 'execution time(s)'
    p = VBox(p1, p2)
    output_file('load.html')
    save(p)

plot()
