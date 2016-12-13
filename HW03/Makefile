CC			:= gcc
CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
MPICC			:= mpicc
PFLAGS			:= -lpthread

IN1 = In_48_640
IN2 = In_1000_100000
N1 = 48
N2 = 1000
IN = $(IN1)
OUT = out.txt
N = $(N1)
SRC = 4

all: SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async

SSSP_Pthread: SSSP_Pthread.c
	$(CC) $? -o $@ $(CCFLAGS) $(PFLAGS)
SSSP_MPI_sync: SSSP_MPI_sync.c
	$(MPICC) -o $@ $? $(CCFLAGS)
SSSP_MPI_async: SSSP_MPI_async.c
	$(MPICC) -o $@ $? $(CCFLAGS)
run:
	#./SSSP_Pthread 2 $(IN) $(OUT) $(SRC)
	mpiexec -n $(N) ./SSSP_MPI_async 4 $(IN) $(OUT) $(SRC)
	./judge_hw3 $(IN) $(OUT) $(SRC)
clean:
	rm -f SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async $(OUT)
