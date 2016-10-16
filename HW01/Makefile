NUM = 150
BASIC = HW1_104062703_basic
ADV = HW1_104062703_advanced
all:
	make gen
	make run
	make check
	make clear
run:
	mpicc -o $(ADV) $(ADV).c
	mpirun -n 10 ./$(ADV) $(NUM) test1.in rst.out
gen:
	python gen_testcase.py -a gen -n $(NUM) -f test1
check:
	python gen_testcase.py -a check -n $(NUM) -f test1
clear:
	rm -f $(BASIC) $(ADV) *.in *.out
