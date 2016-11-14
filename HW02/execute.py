import os
import time

NAME = ['MS_MPI_static',
        'MS_MPI_dynamic',
        'MS_OpenMP_static',
        'MS_OpenMP_dynamic',
        'MS_Hybrid_static',
        'MS_Hybrid_dynamic']
FILE = '#PBS -N HYBRID\n#PBS -r n\n#PBS -l nodes={}:ppn={}\n#PBS -l walltime=00:05:00\n#PBS -o {}\ncd $PBS_O_WORKDIR\nexport MV2_ENABLE_AFFINITY=0\n{}'

CMD = 'time mpiexec -ppn {} ./{} {} -2 2 -2 2 1000 1000 disable'

def test(name):
    os.system('make {}'.format(name))
    sent, count = 0, 0
    
    for node in [1, 2, 3, 4]:
        for proc in [1, 2, 4, 8, 12]:
            job = FILE.format(node, proc, '{}_{}_{}.txt'.format(name, node, proc), CMD.format(node, name, proc))
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
                for f in os.listdir('../'):
                    if 'HYBRID.e' in f:
                        count += 1
                time.sleep(5)
            os.system('qsub {}'.format(jobname))
            sent += 1
    # collect into 1 file
    cts = ''
    for f in sorted(os.listdir('../')):
        if 'HYBRID' in f:
            with open(f, 'r') as fin:
                data = fin.read()
            cts = '{}\n{}'.format(cts, data)
    with open('{}.txt'.format(name), 'w+') as f:
        f.write(cts)
    os.system('rm HYBRID.* job_*')
    os.system('make clean')
for name in NAME:
    test(name)
