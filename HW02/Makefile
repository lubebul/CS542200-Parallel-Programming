CC			:= gcc
CXX			:= g++

MPICC			:= mpicc
MPICXX			:= mpic++

CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
CXXFLAGS		:= -O3 -march=native -Wall -std=c++0x
LDFLAGS			:= -lX11 -fopenmp -lm 

all: MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic MS_seq

MS_MPI_static: MS_MPI_static.c
	$(MPICC) $(CCFLAGS) -o $@ $? $(LDFLAGS)
MS_MPI_dynamic: MS_MPI_dynamic.c
	$(MPICC) $(CCFLAGS) -o $@ $? $(LDFLAGS)
MS_OpenMP_static: MS_OpenMP_static.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)
MS_OpenMP_dynamic: MS_OpenMP_dynamic.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)
MS_Hybrid_static: MS_Hybrid_static.c
	$(MPICC) $(CCFLAGS) -o $@ $? $(LDFLAGS)
MS_Hybrid_dynamic: MS_Hybrid_dynamic.c
	$(MPICC) $(CCFLAGS) -o $@ $? $(LDFLAGS)
MS_seq: MS_seq.c
	$(CC) $(CCFLAGS) -o $@ $? $(LDFLAGS)

run:
	mpiexec -ppn 2 ./MS_Hybrid_dynamic 12 -2 2 -2 2 800 800 disable
clean:
	rm -f MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic MS_seq
