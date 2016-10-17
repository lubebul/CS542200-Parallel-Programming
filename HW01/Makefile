NUM = 300
P = 6
C = test1.out
I = test1.in

BASIC = HW1_104062703_basic
ADV = HW1_104062703_advanced
all:
	make gen
	make run
	make check
	make clear
run:
	mpicc -o $(BASIC) $(BASIC).c
	mpirun -n $(P) ./$(BASIC) $(NUM) $(I) rst.out
gen:
	python gen_testcase.py -a gen -n $(NUM) -f test1
check:
	python gen_testcase.py -a check -n $(NUM) -f $(C) -c rst
clear:
	rm -f $(BASIC) $(ADV) *.in *.out
