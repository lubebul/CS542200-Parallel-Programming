import os
import time

NAME = ['HW4_cuda']
EXE = './{} {} {} {}'

def check(name, fin, fout, ans, b):
    os.system('make {}'.format(name))
    st = time.time()
    os.system(EXE.format(name, fin, fout, b))
    ed = time.time()
    os.system('cmp {} {}'.format(fout, ans))
    os.system('make clean')
    return ed-st

for i in range(1,6):
    t = check(NAME[0], 'testcase/in{}'.format(i), 'out', 'testcase/ans{}'.format(i), 32)
    print('{} takes {}'.format('in{}'.format(i), t))