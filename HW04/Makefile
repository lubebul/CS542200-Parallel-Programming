CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++0x

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_20 -O3 -D_FORCE_INLINES

OMPFLAGS                := -fopenmp
MPILIBS                 := -I/opt/mpich-3.2-gcc/include -L/opt/mpich-3.2-gcc/lib -lmpich

EXES                    := block_FW seq_FW HW4_cuda HW4_openmp HW4_mpi


BASE = block_FW
EXE = HW4_cuda
I = 5
IN = testcase/in$(I)
OUT = out
GT = testcase/ans$(I)
B = 32

all: $(EXES)

clean:
	rm -f $(EXES) $(OUT)

block_FW: block_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

seq_FW: seq_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

HW4_cuda: HW4_104062703_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_openmp: HW4_104062703_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_mpi: HW4_104062703_mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?
run:
	make $(BASE)
	make $(EXE)
	time ./$(BASE) $(IN) $(OUT) $(B)
	time ./$(EXE) $(IN) $(OUT) $(B)
	cmp $(OUT) $(GT)
