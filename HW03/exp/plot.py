from bokeh.plotting import figure, output_file, save
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.layouts import HBox, VBox
from bokeh.models import CheckboxGroup, CustomJS
import glob
import pandas as pd
import numpy as np
import re

colors = [['greenyellow', 'darkgreen'], ['lightblue', 'darkblue'], ['pink', 'red'], ['yellow', 'orange']]

def grabP(fname, size):
    with open(fname, 'r') as f:
        data = f.read()
    lines = data.split('\n')
    ns, comps, syncs, ios = [], [], [], []
    n = 0
    for i in range(11):
        N = int(lines[n])
        n += 1
        for j in range(n, n+N):
            ss = lines[j].split(' ')
            ns.append(N)
            comps.append(float(ss[1]))
            syncs.append(float(ss[2]))
            ios.append(float(ss[3]))
        n += N+1
    return pd.DataFrame({'N':[size for i in range(len(ns))], 'thread':ns, 'compute':comps, 'synchronous':syncs, 'i/o':ios})


def grabMPI(fname, vsize, esize):
    with open(fname, 'r') as f:
        data = f.read()
    lines = data.split('\n')
    ns, comms, comps, syncs, ios, cts = [], [], [], [], [], []
    n = 0
    while n < len(lines):
        s = lines[n].split(' ')
        c = int(s[0][1:])
        ppn = int(s[1][:-1])
        N = c*ppn
        n += 1
        for j in range(n, n+vsize):
            ss = lines[j].split(' ')
            ns.append(N)
            comms.append(float(ss[1]))
            comps.append(float(ss[2]))
            syncs.append(float(ss[3]))
            ios.append(float(ss[4]))
            cts.append(int(ss[5]))
        n += vsize+1
    return pd.DataFrame({'N':[vsize for i in range(len(ns))], 'E':[esize for i in range(len(ns))], '#core*#ppn':ns, 'communicate':comms, 'compute':comps, 'synchronous':syncs, 'i/o':ios, 'count':cts})


def PStrong(df):
    p = figure(title='pthread')
    i = 0
    for n in sorted(list(set(df['N'].values))):
        ndf = df[df['N']==n]
        ths, times = [], []
        for th in sorted(list(set(ndf['thread']))):
            thdf = ndf[ndf['thread']==th]
            ths.append(th)
            times.append(thdf['compute'].mean()+thdf['synchronous'].mean()+thdf['i/o'].mean())
        p.line(ths, times, legend='#vtx={}'.format(n), line_color=colors[i][1], line_width=4)
        p.circle(ths, times, color=colors[i][0], size=10)
        i += 1
    p.xaxis.axis_label = '#thead'
    p.yaxis.axis_label = 'time(s)'
    output_file('pthread_strong.html')
    save(p)


def MPIStrong(df, sas):
    p = figure(title='MPI-{}'.format(sas))
    i = 0
    for n, e in zip([128,256,256,256], [5000, 5000, 10000, 15000]):
        ndf = df[df['N']==n]
        ndf = ndf[ndf['E']==e]
        cps, times = [], []
        for cp in sorted(list(set(ndf['#core*#ppn']))):
            pdf = ndf[ndf['#core*#ppn']==cp]
            cps.append(cp)
            times.append(pdf['communicate'].mean()+pdf['compute'].mean()+pdf['synchronous'].mean()+pdf['i/o'].mean())
        p.line(cps, times, legend='#vtx={},#edge={}'.format(n,e), line_color=colors[i][1], line_width=4)
        p.circle(cps, times, color=colors[i][0], size=10)
        i += 1
    p.xaxis.axis_label = '#core x #ppn'
    p.yaxis.axis_label = 'time(s)'
    output_file('MPI_{}_strong.html'.format(sas))
    save(p)


def PLoad(df):
    i = 0
    ps = {'compute':None, 'synchronous':None, 'i/o':None}
    for n in sorted(list(set(df['N'].values))):
        ndf = df[df['N']==n]
        ths, times = [], []
        ts = []
        th = 128
        thdf = ndf[ndf['thread']==th]
        for term in ['compute', 'synchronous', 'i/o']:
            values = list(reversed(sorted(thdf[term].values)))
            t = ps[term]
            t = figure(title='{}'.format(term)) if t is None else t
            t.line(range(1, len(values)+1), values, legend='#vtx={}'.format(n), line_color=colors[i][1], line_width=2)
            t.circle(range(1, len(values)+1), values, color=colors[i][0], size=5)
            t.xaxis.axis_label = 'thead-id'
            t.yaxis.axis_label = 'time(s)'
            ps[term] = t
        i += 1
    p = VBox(*(ps.values()))
    output_file('pthread_load.html')
    save(p)


def MPILoad(df, sas):
    i = 0
    ps = {'communicate':None, 'compute':None, 'synchronous':None, 'i/o':None}
    for n, e in zip([128,256,256,256], [5000, 5000, 10000, 15000]):
        ndf = df[df['N']==n]
        ndf = ndf[ndf['E']==e]
        if ndf.shape[0] == 0:
            continue
        cp = list(sorted(ndf['#core*#ppn'].values))[-1]
        thdf = ndf[ndf['#core*#ppn']==cp]
        for term in ['communicate', 'compute', 'synchronous', 'i/o']:
            values = list(reversed(sorted(thdf[term].values)))
            t = ps[term]
            t = figure(title='{}'.format(term)) if t is None else t
            t.line(range(1, len(values)+1), values, legend='#vtx={}, #edge={}'.format(n,e), line_color=colors[i][1], line_width=2)
            t.circle(range(1, len(values)+1), values, color=colors[i][0], size=5)
            t.xaxis.axis_label = 'process-id'
            t.yaxis.axis_label = 'time(s)'
            ps[term] = t
        i += 1
    p = VBox(*(ps.values()))
    output_file('MPI_{}_load.html'.format(sas))
    save(p)

Nsize = [128, 256, 1024, 2048, 256, 256]
Esize = [5000, 10000, 100000, 200000, 5000, 15000]
"""
#PThread
df = None
for p in glob.glob('pthread_*.out'):
    i = int(p.split('.')[0].split('_')[1])
    tdf = grabP(p, Nsize[i])
    df = tdf if df is None else pd.concat([df, tdf])
PStrong(df)
PLoad(df)
df.to_csv('pthread.csv', ignore_index=True)
"""
#MPI
for sas in ['sync', 'async']:
    df = None
    for i in [4,5]:
        p = '{}_SSSP_MPI_{}.out'.format(i, sas)
        tdf = grabMPI(p, Nsize[i], Esize[i])
        df = tdf if df is None else pd.concat([df, tdf])
    df.to_csv('MPI_{}.csv'.format(sas))
    MPIStrong(df, sas)
    MPILoad(df, sas)
