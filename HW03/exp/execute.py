import os
import time

NAME = ['SSSP_Pthread', 'SSSP_MPI_sync', 'SSSP_MPI_async']
EXE = ['./{} {} {} {} {}', 'mpiexec -n {1} ./{0} {1} {2} {3} {4}']
FS = ['In_256_10000', 'In_512_20000', 'In_1024_100000', 'In_2048_200000', 'In_256_5000', 'In_256_15000']
PS = [256, 512, 1024, 2048, 256, 256]
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


def testMPI(name, idx):
    FILE = '#PBS -N batch\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:30:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    sent = 0
    # fixed core
    NP = [(1,1), (1,2), (1,4), (1,8), (1,12), (2,12), (3,12), (4,12)]
    for node, ppn in NP:
        CMD = EXE[1].format(name, PS[idx], FS[idx], OUT, 4)
        fname = 'job_{}_{}.sh'.format(node, ppn)
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
    cts = ''
    for node, ppn in NP:
        with open('mpi_{}_{}.txt'.format(node, ppn), 'r') as fin:
            data = fin.read()
        cts = '{}\n[{} {}]\n{}'.format(cts, node, ppn, data)
    with open('{}_{}.out'.format(idx, name), 'w+') as f:
        f.write(cts)
    os.system('rm batch.* *.txt *.sh')
def testPthread(idx):
    FILE = '#PBS -N batch\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:10:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
    sent = 0
    # fixed core
    node, ppn = 1, 12
    ps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for i in ps:
        CMD = EXE[0].format(NAME[0], i, FS[idx], OUT, 4)
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
    with open('pthread_{}.out'.format(idx), 'w+') as f:
        f.write(cts)
    os.system('rm batch.* *.txt *.sh')

os.system('make')
testMPI(NAME[1], 4)
testMPI(NAME[1], 5)
testMPI(NAME[2], 4)
testMPI(NAME[2], 5)
os.system('make clean')
