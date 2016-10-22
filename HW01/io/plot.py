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
    ps, ios = [], []
    for fname in fs:
        ps.append(int(fname.split('.')[0].split('_')[2]))
        with open(fname, 'r') as f:
            data = f.read()
        ios.append(data)
    df = pd.DataFrame({'p':ps, 'io':ios})
    df = df.sort_values(['p'], ascending=[1])
    return df

def plot(seq, mpi, title): 
    p = figure(title=title)
    p.line(seq['p'].values, seq['io'], legend='sequential - I/O', line_color='green', line_width=2)
    p.circle(seq['p'].values, seq['io'], color='green', size=4)
    p.line(mpi['p'].values, mpi['io'], legend='MPI - I/O', line_color='blue', line_width=2)
    p.circle(mpi['p'].values, mpi['io'], color='blue', size=4)
    p.xaxis.axis_label = '#proc'
    p.yaxis.axis_label = 'time(s)'
    
    output_file('{}.html'.format(title))
    save(p)

seq = grab_single('seq')
mpi = grab_single('mpi')
plot(seq, mpi, 'N=1,000,000,000')
