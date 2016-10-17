NUM = 15
C = ../testcase/sorted2
I = ../testcase/testcase2
BASIC = HW1_104062703_basic
ADV = HW1_104062703_advanced
all:
	make run
	make check
	make clear
run:
	mpicc -o $(ADV) $(ADV).c
	mpirun -n 12 ./$(ADV) $(NUM) $(I) rst.out
gen:
	python gen_testcase.py -a gen -n $(NUM) -f test1
check:
	python gen_testcase.py -a check -n $(NUM) -f $(C) -c rst
clear:
	rm -f $(BASIC) $(ADV) *.in *.out
