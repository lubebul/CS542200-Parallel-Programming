CC			:= gcc
CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
MPICC			:= mpicc
PFLAGS			:= -lpthread

IN = In_1000_5000
OUT = out.txt
SRC = 4

# SSSP_MPI_sync SSSP_MPI_async
all: SSSP_Pthread

SSSP_Pthread: SSSP_Pthread.c
	$(CC) $? -o $@ $(CCFLAGS) $(PFLAGS)
SSSP_MPI_sync: SSSP_MPI_sync.c
	$(MPICC) $(CCFLAGS) -o $@ $?
SSSP_MPI_async: SSSP_MPI_async.c
	$(MPICC) $(CCFLAGS) -o $@ $?
run:
	./SSSP_Pthread 4 $(IN) $(OUT) $(SRC)
	./judge_hw3 $(IN) $(OUT) $(SRC)
clean:
	rm -f SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async $(OUT)
