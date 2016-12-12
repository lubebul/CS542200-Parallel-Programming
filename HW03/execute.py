import os
import time

NAME = ['SSSP_Pthread', 'SSSP_MPI_sync', 'SSSP_MPI_async']
EXE = ['./{} {} {} {} {}', 'mpiexec -n {1} ./{0} {1} {2} {3} {4}']
FS = ['In_48_640','In_1000_100000']
PS = [48, 100, 100]
OUT = 'out.txt'


def check(name, exe, pn):
    os.system('make')
    for i in range(1,pn+1):
        os.system(exe.format(name, pn, FS[0], OUT, i))
	os.system('./judge_hw3 {} {} {} > tmp'.format(FS[0], OUT, i))
        suc = open('tmp', 'r').read()
        print(suc)
        if 'Success' not in suc:
            print('Failed {}'.format(i))
            break
    os.system('make clean')


def testMPI(name):
    FILE = '#PBS -N batch\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:10:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    sent = 0
    # fixed core
    NP = [(1,1), (1,2), (1,4), (1,8), (1,12), (2,12), (3,12), (4,12)]
    for node, ppn in NP:
        CMD = EXE[0].format(NAME[0], PS[0], FS[0], OUT, 4)
        fname = 'job_p_{}.sh'.format(i)
        with open(fname, 'w+') as f:
            f.write(FILE.format(node, ppn, 'mpi_{}_{}.txt'.format(node, ppn), CMD))
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if 'batch.e' in f:
                count += 1
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if 'batch.e' in f:
                    count += 1
	    print('count={}, sent={}'.format(count, sent))
	    time.sleep(1)
        os.system('qsub {}'.format(fname))
	sent += 1
    while sent != count:
	count = 0
	for f in os.listdir('.'):
	    if 'batch.e' in f:
		count += 1
	print('count={}, sent={}'.format(count, sent))
	time.sleep(1)
    time.sleep(10)
    # collect results
    for node, ppn in NP:
        with open('mpi_{}_{}.txt'.format(node, ppn), 'r') as fin:
            data = fin.read()
        cts = '{}\n[{} {}] {}'.format(cts, node, ppn, data)
    with open('{}.out'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm batch.* *.txt *.sh')
def testPthread():
    FILE = '#PBS -N batch\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:10:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    sent = 0
    # fixed core
    node, ppn = 1, 12
    ps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for i in ps:
        CMD = EXE[1].format(name, i, FS[1], OUT, 4)
        fname = 'job_p_{}.sh'.format(i)
        with open(fname, 'w+') as f:
            f.write(FILE.format(node, ppn, 'p_{}.txt'.format(i), CMD))
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if 'batch.e' in f:
                count += 1
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if 'batch.e' in f:
                    count += 1
	    print('count={}, sent={}'.format(count, sent))
	    time.sleep(1)
        os.system('qsub {}'.format(fname))
	sent += 1
    while sent != count:
	count = 0
	for f in os.listdir('.'):
	    if 'batch.e' in f:
		count += 1
	print('count={}, sent={}'.format(count, sent))
	time.sleep(1)
    time.sleep(10)
    # collect results
    cts = 'node={}, ppn={}'.format(node, ppn)
    for i in ps:
        with open('p_{}.txt'.format(i), 'r') as fin:
            data = fin.read()
        cts = '{}\n[{}] {}'.format(cts, i, data)
    with open('pthread_p.out', 'w+') as f:
        f.write(cts)
    os.system('rm batch.* *.txt *.sh')

os.system('make')
#testPthread()
testMPI(NAME[1])
testMPI(NAME[2])
os.system('make clean')
