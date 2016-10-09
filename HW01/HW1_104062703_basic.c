#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

void openFiles(char* in, char* out);
void closeFiles(void);
void printNums(int len, int* nums);
void swap (int *nums, int i, int j);
int* oddEvenSort(int len, int *nums);

int main (int argc, char** argv) {
  int rank, size;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("size = %d, rank = %d\n", size, rank);
  MPI_File *fin, *fout;
  // dealing with I/Os
  assert(("Usage: ./HW_104062703_basic N in-file out-file\n") && argc == 4);
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDWR, MPI_INFO_NULL, &fin);
  MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR, MPI_INFO_NULL, &fout);
  // read number array
  int N = atoi(argv[1]); int *nums = malloc(N*sizeof(int));
  MPI_Status suc;
  MPI_File_read(fin, nums, N, MPI_INT, &suc);
  printNums(N, nums);
  nums = oddEvenSort(N, nums);
  printNums(N, nums);
  // write sorted array
  MPI_File_write(fout, nums, N, MPI_INT, &suc);
  MPI_File_close(fin); MPI_File_close(fout);
  MPI_Finalize ();
  return 0;
}

void printNums(int len, int* nums) {
  int i;
  for (i=0; i<len; i++) printf("****");
  printf("\n");
  for (i=0; i<len; i++) {
    if (i>0) printf(" "); printf("%d", nums[i]);
  }
  printf("\n");
  for (i=0; i<len; i++) printf("****");
  printf("\n");
}
void swap (int *nums, int i, int j) {
  int tmp = nums[i];
  nums[i] = nums[j];
  nums[j] = tmp;
}
int* oddEvenSort(int len, int *nums) {
  int sorted = 0; int i;
  while (!sorted) {
    sorted = 1;
    for (i=1; i<len-1; i+=2) if (nums[i] > nums[i+1]) { swap(nums, i, i+1); sorted = 0; }
    for (i=0; i<len-1; i+=2) if (nums[i] > nums[i+1]) { swap(nums, i, i+1); sorted = 0; }
  }
  return nums;
}
