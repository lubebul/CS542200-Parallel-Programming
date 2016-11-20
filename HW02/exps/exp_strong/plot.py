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
    p = figure(title=title)
    colors = [['greenyellow', 'darkgreen'], ['lightblue', 'darkblue'], ['pink', 'red']]
    for c0, ver in zip(range(3), ['MPI', 'OpenMP', 'Hybrid']):
        for c1, tp in zip(range(2), ['static', 'dynamic']):
            with open('MS_{}_{}_strong.out'.format(ver, tp), 'r') as f:
                data = f.read()
            a1 = [int(x) for x in re.findall('(\d*) \d* .*', data)]
            a2 = [int(x) for x in re.findall('\d* (\d*) .*', data)]
            xs = [x*y for x,y in zip(a1,a2)]
            times = [float(x) for x in re.findall('\d* \d* (.*)', data)]
            p.line(xs, times, legend='{}-{}'.format(ver, tp), line_color=colors[c0][c1], line_width=2)
            p.circle(xs, times, color=colors[c0][c1], size=4)
    p.xaxis.axis_label = '#proc'
    p.yaxis.axis_label = 'time(s)'
    output_file('strong.html')
    save(p)

plot('strong-scalability: 1000*1000')
