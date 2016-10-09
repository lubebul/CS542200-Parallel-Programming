all:
	make gen
	make run
run:
	mpicc -o HW1_104062703_basic HW1_104062703_basic.c
	mpirun -n 15 ./HW1_104062703_basic 15 test1.in rst.out
gen:
	python gen_testcase.py -a gen -n 15 -f test1
check:
	python gen_testcase.py -a check -n 15 -f test1
