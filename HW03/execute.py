import os
import time

NAME = ['SSSP_Pthread', 'SSSP_MPI_sync', 'SSSP_MPI_async']
EXE = ['./{} {} {} {} {}', 'mpiexec -n {1} ./{0} {1} {2} {3} {4}']
FS = ['In_48_60', 'In_100_4500', 'In_1000_100000']
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


def testStrong(name):
    FILE = '#PBS -N batch\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'
