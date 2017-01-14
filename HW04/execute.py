import os
import time

BN = [8,16,32]
FS = ['I100', 'I500', 'I1000', 'I2500', 'I5000']

def run(name, exe, gn):
    FILE = '#PBS -N CUDA_JOB\n#PBS -r n\n#PBS -l nodes=1:ppn={0}:gpus={0}:exclusive_process\n#PBS -l walltime=00:10:00\n#PBS -o {1}\ncd $PBS_O_WORKDIR\n{2}'
    sent = 0
    # fixed core
    for f in FS:
        for bn in BN:
            fname = 'job_{}_{}.sh'.format(f, bn)
            fout = '{}_{}_{}'.format(name[:-4], f, bn)
            CMD = exe.format(name, f, '{}.out'.format(fout), bn)
            with open(fname, 'w+') as f:
                f.write(FILE.format(gn, '{}.txt'.format(fout), CMD))
        # check current executed jobs
        count = 0
        for f in os.listdir('../'):
            if '.out' in f:
                count += 1
        count /= 2
        while sent-count > 7:
            count = 0
            for f in os.listdir('.'):
                if '.out' in f:
                    count += 1
            count /= 2
            print('count={}, sent={}'.format(count, sent))
            time.sleep(1)
        os.system('qsub {}'.format(fname))
        sent += 1
    while sent != count:
        for f in os.listdir('.'):
            if '.out' in f:
                count += 1
        print('count={}, sent={}'.format(count, sent))
        time.sleep(1)
    time.sleep(10)
    # collect results
    cts = ''
    for f in FS:
        for bn in BN:
            with open('{}_{}_{}'.format(name[:-4], f, bn), 'r') as fin:
                data = fin.read()
            cts = '{}\n[{} {}]\n{}'.format(cts, f, bn, data)
    with open('{}.tot'.format(name[:-4]), 'w+') as f:
        f.write(cts)
    os.system('rm *.out *.sh')

NAME = ['block_FW.exe', 'HW4_cuda.exe', 'HW4_openmp.exe', 'HW4_mpi.exe', 'HW4_cuda_noS.exe', 'HW4_openmp_noS.exe']
EXEs = ['./{} {} {} {}', 'mpirun ./{} {} {} {}']
PEXE = [EXEs[0], EXEs[0], EXEs[0], EXEs[1], EXEs[0], EXEs[0]]
PG = [1,1,2,2,1,2]

os.system('make')
for i in range(6):
    run(NAME[i], PEXE[i], PG[i])
os.system('make clean')