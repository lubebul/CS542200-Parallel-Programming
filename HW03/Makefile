CC			:= gcc
CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
MPICC			:= mpicc
PFLAGS			:= -lpthread

IN1 = In_5_7
IN2 = In_1000_5000
N1 = 5
N2 = 1000
IN = $(IN2)
OUT = out.txt
N = $(N2)
SRC = 4

# SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async
all: SSSP_MPI_sync

SSSP_Pthread: SSSP_Pthread.c
	$(CC) $? -o $@ $(CCFLAGS) $(PFLAGS)
SSSP_MPI_sync: SSSP_MPI_sync.c
	$(MPICC) -o $@ $? $(CCFLAGS)
SSSP_MPI_async: SSSP_MPI_async.c
	$(MPICC) -o $@ $? $(CCFLAGS)
run:
	mpiexec -n $(N) ./SSSP_MPI_sync 4 $(IN) $(OUT) $(SRC)
	./judge_hw3 $(IN) $(OUT) $(SRC)
clean:
	rm -f SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async $(OUT)
