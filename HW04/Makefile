CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++0x

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_20 -O3 -D_FORCE_INLINES

OMPFLAGS                := -fopenmp
MPILIBS                 := -I/opt/mpich-3.2-gcc/include -L/opt/mpich-3.2-gcc/lib -lmpich
EXES                    := block_FW.exe seq_FW.exe HW4_cuda.exe HW4_openmp.exe HW4_mpi.exe HW4_openmp_noS.exe HW4_cuda_noS.exe


BASE = block_FW.exe
EXE = HW4_cuda_noS.exe
I = 5
IN = testcase/in$(I)
OUT = out
GT = testcase/ans$(I)
B = 32

all: $(EXES)

clean:
	rm -f $(EXES) $(OUT) tmp*

block_FW.exe: block_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

seq_FW.exe: seq_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

HW4_cuda.exe: HW4_104062703_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_openmp.exe: HW4_104062703_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_mpi.exe: HW4_104062703_mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?

HW4_cuda_noS.exe: HW4_104062703_cuda_noS.cu
	$(NVCC) $(NVFLAGS) -o $@ $?
HW4_openmp_noS.exe: HW4_104062703_openmp_noS.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?
run:
	make $(EXE)
	time ./$(EXE) $(IN) $(OUT) $(B)
	cmp $(OUT) $(GT)
