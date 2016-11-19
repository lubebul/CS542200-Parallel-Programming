import os
import time

NAME = ['MS_MPI_static',
        'MS_MPI_dynamic',
        'MS_OpenMP_static',
        'MS_OpenMP_dynamic',
        'MS_Hybrid_static',
        'MS_Hybrid_dynamic']
FILE = '#PBS -N HYBRID\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'

CMD = 'mpiexec -n {} ./{} {} -2 2 -2 2 1000 1000 disable'

def test(name):
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
    with open('{}.out'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm HYBRID.* *.txt')
    os.system('make clean')
for name in NAME:
    test(name)
