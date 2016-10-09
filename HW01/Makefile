all:
	make gen
	make run
run:
	mpicc -o HW1_104062703_basic HW1_104062703_basic.c
	mpirun -n 4 ./HW1_104062703_basic 10 test1.in rst.out
gen:
	python gen_testcase.py -a gen -n 10 -f test1
check:
	python gen_testcase.py -a check -n 10 -f test1
