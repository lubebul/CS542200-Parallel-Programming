from bokeh.charts import Bar, output_file, save
from bokeh.charts.attributes import cat, color
from bokeh.charts.operations import blend
from bokeh.charts.utils import df_from_json
from bokeh.sampledata.olympics2014 import data
import pandas as pd
import re

def plot_weak(name):
    with open('{}.tot'.format(name), 'r') as f:
        data = f.read()
    Ns = [int(x) for x in re.findall('\[I(\d*) \d*\]\n.* .* .*', data)]
    Bs = [int(x) for x in re.findall('\[I\d* (\d*)\]\n.* .* .*', data)]
    T1s = [float(x) for x in re.findall('\[I\d* \d*\]\n(.*) .* .*', data)]
    T2s = [float(x) for x in re.findall('\[I\d* \d*\]\n.* (.*) .*', data)]
    T3s = [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* (.*)', data)]
    """
    if 'cuda' not in name:
        Ns += Ns
        Bs += Bs
        T1s += [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* .*\n(.*) .* .*', data)]
        T2s += [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* .*\n.* (.*) .*', data)]
        T3s += [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* .*\n.* .* (.*)', data)]
    """
    df = pd.DataFrame({'N':Ns, 'B':Bs, 'computation':T1s, 'communication':T2s, 'memcpy':T3s})
    bar = Bar(df, label=['N','B'], values=blend('computation','communication','memcpy', name='times', labels_name='time'), xlabel='#node', ylabel='total time(s)', stack=cat(columns=['time'], sort=False), title='Weak Scalability')
    output_file('weak_{}.html'.format(name))
    save(bar)
def plot_opt(name):
    with open('{}.txt'.format(name), 'r') as f:
        data = f.read()
    N = int(re.findall('\[I(\d*) \d*\]\n.* .* .*', data)[0])
    B = int(re.findall('\[I\d* (\d*)\]\n.* .* .*', data)[0])
    TP = ['with', 'without']
    T1 = [float(x) for x in re.findall('\[I\d* \d*\]\n(.*) .* .*', data)]
    T2 = [float(x) for x in re.findall('\[I\d* \d*\]\n.* (.*) .*', data)]
    T3 = [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* (.*)', data)]
    T1 += [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* .*\n(.*) .* .*', data)]
    T2 += [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* .*\n.* (.*) .*', data)]
    T3 += [float(x) for x in re.findall('\[I\d* \d*\]\n.* .* .*\n.* .* (.*)', data)]
    df = pd.DataFrame({'type':TP, 'computation':T1, 'communication':T2, 'memcpy':T3})
    bar = Bar(df, label=['type'], values=blend('computation','communication','memcpy', name='times', labels_name='time'), xlabel='comparison', ylabel='total time(s)', stack=cat(columns=['time'], sort=False), title='{}'.format(name))
    output_file('{}.html'.format(name), title='N={}, B={}'.format(N,B))
    save(bar)
for name in ['HW4_cuda', 'HW4_openmp', 'HW4_mpi']:
    plot_weak(name)
plot_opt('share-memory')
plot_opt('streaming')
