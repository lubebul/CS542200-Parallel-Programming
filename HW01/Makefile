NUM = 255
all:
	make gen
	make run
	make check
	make clear
run:
	mpicc -o HW1_104062703_basic HW1_104062703_basic.c
	mpirun -n 18 ./HW1_104062703_basic $(NUM) test1.in rst.out
gen:
	python gen_testcase.py -a gen -n $(NUM) -f test1
check:
	python gen_testcase.py -a check -n $(NUM) -f test1
clear:
	rm HW1_104062703_basic *.in *.out
