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

def plot_break(df, title): 
    p1 = figure(title=title)
    p1.line(df['p'].values, df['tot'], legend='total', line_color='black', line_width=1)
    p1.line(df['p'].values, df['cmp'], legend='computation', line_color='red', line_width=4)
    p1.line(df['p'].values, df['comm'], legend='communication', line_color='blue', line_width=4)
    p1.line(df['p'].values, df['io'], legend='I/O', line_color='green', line_width=1)
    # strong scale
    s = [float(df['tot'].values[0])/float(x) for x in df['p'].values]
    c = [float(df['cmp'].values[0])/float(x) for x in df['p'].values]
    p1.line(df['p'].values, s, legend='ideal: total', line_color='black', line_width=1, line_dash='dashed')
    p1.line(df['p'].values, c, legend='ideal: computation', line_color='red', line_width=1, line_dash='dashed')
    
    p1.circle(df['p'].values, df['tot'], color='black', size=2)
    p1.circle(df['p'].values, df['cmp'], color='red', size=6)
    p1.circle(df['p'].values, df['comm'], color='blue', size=6)
    p1.circle(df['p'].values, df['io'], color='green', size=2)
    p1.xaxis.axis_label = '#proc'
    p1.yaxis.axis_label = 'time(s)'

    p2 = figure(title=title)
    sp = [float(df['tot'].values[0])/float(df['tot'].values[i]) for i in range(df.shape[0])]
    p2.line(df['p'].values, sp, legend='speedup', line_color='blue', line_width=3)
    # strong scale
    p2.line(df['p'].values, df['p'].values, legend='ideal: speedup', line_color='black', line_width=2, line_dash='dashed')
    p2.circle(df['p'].values, sp, color='blue', size=6)
    p2.xaxis.axis_label = '#proc'
    p2.yaxis.axis_label = 'speedup-factor'

    p = HBox(p1, p2)
    output_file('{}.html'.format(title))
    save(p)

df = grab_single('adv')
plot_break(df, 'N=4,000,000,000')
