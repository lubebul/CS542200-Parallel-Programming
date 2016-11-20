import os
import time

NAME = ['MS_MPI_static',
        'MS_MPI_dynamic',
        'MS_OpenMP_static',
        'MS_OpenMP_dynamic',
        'MS_Hybrid_static',
        'MS_Hybrid_dynamic']
def testStrong(name):
    FILE = '#PBS -N HYBRID\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    CMD = 'mpiexec -n {} ./{} {} -2 2 -2 2 1000 1000 disable'
    os.system('make {}'.format(name))
    sent, count = 0, 0
    PAIR = [(1,1), (1,2), (1,4), (1,8), (1,12), (2,12), (3,12), (4,12)]
    NP = PAIR if 'OpenMP' not in name else [(1,1), (1,2), (1,4), (1,8), (1,12), (1,24), (1,36), (1,48)]
    NP = PAIR if 'MPI' not in name else [(1,1), (2,1), (4,1), (8,1), (12,1), (24,1), (36,1), (48,1)]
    for (node, proc), (pnode, pproc) in zip(PAIR, NP):
        job = FILE.format(node, proc, '{}_{}_{}.txt'.format(name, node, proc), CMD.format(pnode, name, pproc))
        jobname = 'job_{}_{}_{}.txt'.format(name, node, proc)
        with open(jobname, 'w+') as f:
            f.write(job)
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if 'HYBRID.e' in f:
                count += 1
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if 'HYBRID.e' in f:
                    count += 1
	    print('count={}, sent={}'.format(count, sent))
	    time.sleep(1)
        os.system('qsub {}'.format(jobname))
        sent += 1
    while sent != count:
	count = 0
	for f in os.listdir('.'):
	    if 'HYBRID.e' in f:
		count += 1
	print('count={}, sent={}'.format(count, sent))
	time.sleep(1)
    time.sleep(10)
    # collect into 1 file
    cts = ''
    for node, proc in PAIR:
        with open('{}_{}_{}.txt'.format(name, node, proc), 'r') as fin:
            data = fin.read()
        cts = '{}\n{}'.format(cts, data)
    with open('{}_strong.out'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm HYBRID.* *.txt')
    os.system('make clean')
def testWeak(name):
    FILE = '#PBS -N HYBRID\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    CMD = 'mpiexec -n {} ./{} {} -2 2 -2 2 {} {} disable'
    os.system('make {}'.format(name))
    sent, count = 0, 0
    if 'Hybrid' in name:
        node, proc = 4, 2
        pnode, pproc = 4, 2
    elif 'OpenMP' in name:
        node, proc = 1, 8
        pnode, pproc = 1, 8
    elif 'MPI' in name:
        node, proc = 1, 8
        pnode, pproc = 8, 1
    Ns = [10, 20, 40, 80, 160, 320, 640, 1280]
    for N in Ns:
        job = FILE.format(node, proc, '{}_{}.txt'.format(name, N), CMD.format(pnode, name, pproc, N, N))
        jobname = 'job_{}_{}.txt'.format(name, N)
        with open(jobname, 'w+') as f:
            f.write(job)
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if 'HYBRID.e' in f:
                count += 1
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if 'HYBRID.e' in f:
                    count += 1
	    print('count={}, sent={}'.format(count, sent))
	    time.sleep(1)
        os.system('qsub {}'.format(jobname))
        sent += 1
    while sent != count:
	count = 0
	for f in os.listdir('.'):
	    if 'HYBRID.e' in f:
		count += 1
	print('count={}, sent={}'.format(count, sent))
	time.sleep(1)
    time.sleep(10)
    # collect into 1 file
    cts = ''
    for N in Ns:
        with open('{}_{}.txt'.format(name, N), 'r') as fin:
            data = fin.read()
        cts = '{}\n{}: {}'.format(cts, N, data)
    with open('{}_weak.out'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm HYBRID.* *.txt')
    os.system('make clean')
def testLoad(name):
    FILE = '#PBS -N HYBRID\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    CMD = 'mpiexec -n {} ./{} {} -2 2 -2 2 {} {} disable'
    os.system('make {}'.format(name))
    sent, count = 0, 0
    if 'Hybrid' in name:
        node, proc = 4, 2
        pnode, pproc = 4, 2
    elif 'OpenMP' in name:
        node, proc = 1, 8
        pnode, pproc = 1, 8
    elif 'MPI' in name:
        node, proc = 1, 8
        pnode, pproc = 8, 1
    Ns = [1280]
    for N in Ns:
        job = FILE.format(node, proc, '{}_{}.txt'.format(name, N), CMD.format(pnode, name, pproc, N, N))
        jobname = 'job_{}_{}.txt'.format(name, N)
        with open(jobname, 'w+') as f:
            f.write(job)
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if 'HYBRID.e' in f:
                count += 1
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if 'HYBRID.e' in f:
                    count += 1
	    print('count={}, sent={}'.format(count, sent))
	    time.sleep(1)
        os.system('qsub {}'.format(jobname))
        sent += 1
    while sent != count:
	count = 0
	for f in os.listdir('.'):
	    if 'HYBRID.e' in f:
		count += 1
	print('count={}, sent={}'.format(count, sent))
	time.sleep(1)
    time.sleep(10)
    # collect into 1 file
    cts = ''
    for N in Ns:
        with open('{}_{}.txt'.format(name, N), 'r') as fin:
            data = fin.read()
        cts = '{}\n{}'.format(cts, data)
    with open('{}_load.out'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm HYBRID.* *.txt')
    os.system('make clean')
def testBest(name='MS_Hybrid_dynamic'):
    FILE = '#PBS -N HYBRID\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    CMD = 'mpiexec -n {} ./{} {} -2 2 -2 2 1000 1000 disable'
    os.system('make {}'.format(name))
    sent, count = 0, 0
    PAIR = [(1,12), (2,6), (3,4), (4,3), (6,2), (12,1)]
    for (node, proc), (pnode, pproc) in zip(PAIR, NP):
        job = FILE.format(node, proc, '{}_{}_{}.txt'.format(name, node, proc), CMD.format(pnode, name, pproc))
        jobname = 'job_{}_{}_{}.txt'.format(name, node, proc)
        with open(jobname, 'w+') as f:
            f.write(job)
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if 'HYBRID.e' in f:
                count += 1
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if 'HYBRID.e' in f:
                    count += 1
	    print('count={}, sent={}'.format(count, sent))
	    time.sleep(1)
        os.system('qsub {}'.format(jobname))
        sent += 1
    while sent != count:
	count = 0
	for f in os.listdir('.'):
	    if 'HYBRID.e' in f:
		count += 1
	print('count={}, sent={}'.format(count, sent))
	time.sleep(1)
    time.sleep(10)
    # collect into 1 file
    cts = ''
    for node, proc in PAIR:
        with open('{}_{}_{}.txt'.format(name, node, proc), 'r') as fin:
            data = fin.read()
        cts = '{}\n{}'.format(cts, data)
    with open('{}_strong.out'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm HYBRID.* *.txt')
    os.system('make clean')

testBest()
