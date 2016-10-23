from bokeh.plotting import figure, output_file, save
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.layouts import HBox, VBox
from bokeh.charts import BoxPlot
from bokeh.models import CheckboxGroup, CustomJS
import glob
import pandas as pd
import numpy as np


def grab_single(ver):
    fs = glob.glob('{}_mse_[0-9]*.txt'.format(ver))
    ps, cmps, comms, ios, tots = [], [], [], [], []
    for fname in fs:
        ps.append(int(fname.split('.')[0].split('_')[2]))
        with open(fname, 'r') as f:
            data = f.read()
        ns = data.split(' ')
        cmps.append(ns[0])
        comms.append(ns[1])
        ios.append(ns[2])
        tots.append(ns[3])
    df = pd.DataFrame({'p':ps, 'cmp':cmps, 'comm':comms, 'io':ios, 'tot':tots})
    df = df.sort_values(['p'], ascending=[1])
    return df

def plot_break(single, multi, title):
    p = figure(title=title)
    # strong scale
    f = float(multi['cmp'].values[0])/float(multi['tot'].values[0])
    s = [float(single['tot'].values[0])/float(single['tot'].values[i]) for i in range(single.shape[0])]
    m = [float(single['tot'].values[0])/float(multi['tot'].values[i]) for i in range(multi.shape[0])]
    i = [1./((1-f)+f/float(multi['p'].values[i])) for i in range(multi.shape[0])]
    p.line(single['p'].values, s, legend='single node', line_color='green', line_width=2)
    p.line(multi['p'].values, m, legend='multi node', line_color='blue', line_width=2)
    p.line(multi['p'].values, i, legend='maximum speedup', line_color='black', line_width=1, line_dash='dashed')
    
    p.circle(single['p'].values, s, color='green', size=4)
    p.circle(multi['p'].values, m, color='blue', size=4)
    p.xaxis.axis_label = '#proc'
    p.yaxis.axis_label = 'speedup'
    output_file('{}.html'.format(title))
    save(p)

single = grab_single('single/adv')
multi = grab_single('multi/adv')
plot_break(single, multi, 'N=500,000,000')
