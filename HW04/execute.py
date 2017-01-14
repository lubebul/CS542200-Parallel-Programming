import os
import time

BN = [8,16,32]
FS = ['I100', 'I500', 'I1000', 'I2500', 'I5000']

def run(name, exe, gn):
    FILE = '#PBS -N CUDA_JOB\n#PBS -r n\n#PBS -l nodes=1:ppn={0}:gpus={0}:exclusive_process\n#PBS -l walltime=00:10:00\n#PBS -o {1}\ncd $PBS_O_WORKDIR\n{2}'
    sent = 0
    # fixed core
    for ff in FS:
        for bn in BN:
            fname = 'job_{}_{}_{}.sh'.format(name[:-4], ff, bn)
            fout = '{}_{}_{}'.format(name[:-4], ff, bn)
            CMD = exe.format(name, 'exp_test/{}'.format(ff), '{}.out'.format(fout), bn)
            with open(fname, 'w+') as f:
                f.write(FILE.format(gn, '{}.out'.format(fout), CMD))
	    # check current executed jobs
	    count = 0
    	    for f in os.listdir('.'):
        	if 'CUDA_JOB.e' in f:
            	    count += 1
            while sent-count > 3:
		time.sleep(1)
                count = 0
                for f in os.listdir('.'):
                    if 'CUDA_JOB.e' in f:
                        count += 1
                print('count={}, sent={}'.format(count, sent))
            os.system('qsub {}'.format(fname))
            sent += 1
    while sent != count:
	count = 0
        for f in os.listdir('.'):
            if 'CUDA_JOB.e' in f:
                count += 1
        print('[wait all] count={}, sent={}'.format(count, sent))
        time.sleep(1)
    time.sleep(10)
    # collect results
    cts = ''
    for ff in FS:
        for bn in BN:
            with open('{}_{}_{}.out'.format(name[:-4], ff, bn), 'r') as fin:
                data = fin.read()
            cts = '{}\n[{} {}]\n{}'.format(cts, ff, bn, data)
    with open('{}.tot'.format(name[:-4]), 'w+') as f:
        f.write(cts)
    os.system('rm CUDA_* *.out *.sh')

NAME = ['HW4_cuda.exe', 'HW4_openmp.exe', 'HW4_mpi.exe', 'HW4_cuda_noS.exe', 'HW4_openmp_noS.exe']
EXEs = ['./{} {} {} {}', 'mpirun ./{} {} {} {}']
PEXE = [EXEs[0], EXEs[0], EXEs[1], EXEs[0], EXEs[0]]
PG = [1,2,2,1,2]

os.system('make')
for i in range(5):
    run(NAME[i], PEXE[i], PG[i])
os.system('make clean')
