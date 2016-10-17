#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define max(x,y) (x)>(y)?(x):(y)
#define min(x,y) (x)<(y)?(x):(y)
// measure time
double cmptime, commtime, iotime, tot;
int mnode;

int cmp(const void* a, const void* b);
void openFiles(char* in, char* out);
void closeFiles(void);
void printNums(int len, int* nums);
void swap (int *nums, int i, int j);
int* merge(int *A, int *B, int asize, int bsize);
int* seqOddEvenSort(int len, int *nums);
int* parOddEvenSort(int len, int *nums, int rank, int size, int chunk, MPI_Comm W_COMM);
void writeMeasure(double tot, double cmptime, double commtime, double iotime, FILE *fmse, int size);

int main (int argc, char** argv) {
  int rank, size;
  // measure time
  double ss;
  FILE *fmse; cmptime = commtime = iotime = tot = 0.0; mnode = 0; ss = 0;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == mnode) tot = ss = MPI_Wtime();
  MPI_File fin, fout;
  // dealing with I/Os
  assert(("Usage: ./HW_104062703_advanced N in-file out-file\n") && argc == 4);
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
  MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
  // read number array
  int N = atoi(argv[1]); int *nums = malloc(N*sizeof(int));
  MPI_Status suc;
  
  MPI_File_read(fin, nums, N, MPI_INT, &suc);
  if (rank == mnode) iotime += MPI_Wtime()-ss;
  
  if (N < 100) { // too less items
    if (rank == mnode) {
      // computation
      ss = MPI_Wtime();
      qsort(nums, N, sizeof(int), cmp);
      cmptime += MPI_Wtime()-ss;
      // I/O time
      ss = MPI_Wtime();
      MPI_File_write(fout, nums, N, MPI_INT, &suc);
      iotime += MPI_Wtime()-ss;
    }
    // I/O time
    ss = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    commtime += MPI_Wtime()-ss;
    MPI_File_close(&fin); MPI_File_close(&fout);
    if (rank == mnode) writeMeasure(MPI_Wtime()-tot, cmptime, commtime, iotime, fmse, size);
    MPI_Finalize();
    free(nums);
    return 0;
  }

  // communication: extract working groups
  if (rank == mnode) ss = MPI_Wtime();
  MPI_Comm W_COMM = MPI_COMM_WORLD;
  if (size > N/3-1) { // too much workers
    size = N/3-1;
    int i; int ranks[size]; for (i=0;i<size;i++) ranks[i]=i;
    MPI_Group O_Group, W_Group;
    MPI_Comm_group(MPI_COMM_WORLD, &O_Group);
    MPI_Group_incl(O_Group, size, ranks, &W_Group);
    MPI_Comm_create(MPI_COMM_WORLD, W_Group, &W_COMM);
  } else { // items must be divisible
    int chunk = (N+size-1) / size;
    if ((size-1)*chunk >= N) {
      size -= 1;
      int i; int ranks[size]; for (i=0;i<size;i++) ranks[i]=i;
      MPI_Group O_Group, W_Group;
      MPI_Comm_group(MPI_COMM_WORLD, &O_Group);
      MPI_Group_incl(O_Group, size, ranks, &W_Group);
      MPI_Comm_create(MPI_COMM_WORLD, W_Group, &W_COMM);
    }
  }
  if (rank == mnode) commtime += MPI_Wtime()-ss;
  
  int chunk = (N+size-1) / size;
  if (rank < size) nums = parOddEvenSort(N, nums, rank, size, chunk, W_COMM);
  if (rank == mnode) {
    // I/O
    ss = MPI_Wtime();
    MPI_File_write(fout, nums, N, MPI_INT, &suc);
    iotime += MPI_Wtime()-ss;
    writeMeasure(MPI_Wtime()-tot, cmptime, commtime, iotime, fmse, size);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fin); MPI_File_close(&fout);
  free(nums);
  MPI_Finalize();
  return 0;
}

int cmp(const void* a, const void* b) {
  int va = *(int *) a; int vb = *(int *) b;
  return (va > vb) - (va < vb);
}
void printNums(int len, int* nums) {
  int i;
  for (i=0; i<len; i++) printf("****"); printf("\n");
  for (i=0; i<len; i++) {if (i>0) printf(" "); printf("%d", nums[i]);} printf("\n");
  for (i=0; i<len; i++) printf("****"); printf("\n");
}
void swap (int *nums, int i, int j) { int tmp = nums[i];  nums[i] = nums[j]; nums[j] = tmp;}
int* seqOddEvenSort(int len, int *nums) {
  int sorted = 0; int i;
  while (!sorted) {
    sorted = 1;
    for (i=1; i<len-1; i+=2) if (nums[i] > nums[i+1]) { swap(nums, i, i+1); sorted = 0; }
    for (i=0; i<len-1; i+=2) if (nums[i] > nums[i+1]) { swap(nums, i, i+1); sorted = 0;}
  }
  return nums;
}
int* merge(int *A, int *B, int asize, int bsize) {
  int *tmp = (int *)malloc(sizeof(int)*(asize+bsize));
  int a = 0; int b = 0; int i;
  for (i=0; i<asize+bsize; i++) {
    if (a == asize) {tmp[i] = B[b++];}
    else if (b == bsize) {tmp[i] = A[a++];}
    else if (A[a] <= B[b]) {tmp[i] = A[a++];}
    else {tmp[i] = B[b++];}
  }
  return tmp;
}
int* parOddEvenSort(int len, int *nums, int rank, int size, int chunk, MPI_Comm W_COMM) {
  int s = size;
  int st = min(rank*chunk, len-1); int ed = min(st + chunk-1, len-1);
  int i; MPI_Status suc;
  int *tmp_merge = (int *)malloc(sizeof(int)*chunk*2);
  double ss;
  // computation
  if (rank == mnode) ss = MPI_Wtime();
  qsort(&nums[st], ed-st+1, sizeof(int), cmp);
  if (rank == mnode) cmptime += MPI_Wtime()-ss;
  
  while (s--) {
    // odd phase
    if ((rank%2 == 0) && (rank>=1)) {
      // send to sort
      int rsize = ed-st+1;
      // communication
      if (rank == mnode) ss = MPI_Wtime();
      
      MPI_Send(&rsize, 1, MPI_INT, rank-1, 0, W_COMM);
      MPI_Send(&nums[st], ed-st+1, MPI_INT, rank-1, 1, W_COMM);
      // receive sorted
      MPI_Recv(&nums[st], ed-st+1, MPI_INT, rank-1, 2, W_COMM, &suc);
      
      if (rank == mnode) commtime += MPI_Wtime()-ss;
    } else if ((rank%2 == 1) && (rank+1<size)) {
      int rsize;
      // communication
      if (rank == mnode) ss = MPI_Wtime();
      
      MPI_Recv(&rsize, 1, MPI_INT, rank+1, 0, W_COMM, &suc);
      MPI_Recv(&nums[ed+1], rsize, MPI_INT, rank+1, 1, W_COMM, &suc);

      if (rank == mnode) commtime += MPI_Wtime()-ss;

      // computation: merge
      if (rank == mnode) ss = MPI_Wtime();
      tmp_merge = merge(&nums[st], &nums[ed+1], ed-st+1, rsize);
      for (i=0; i<rsize+(ed-st+1); i++) nums[st+i] = tmp_merge[i];
      
      if (rank == mnode) cmptime += MPI_Wtime()-ss;
      // communication
      if (rank == mnode) ss = MPI_Wtime();
      MPI_Send(&nums[ed+1], rsize, MPI_INT, rank+1, 2, W_COMM);
      if (rank == mnode) commtime += MPI_Wtime()-ss;
    }
    
    // even phase
    if ((rank%2 == 1) && (rank>=1)) {
      // send to sort
      int rsize = ed-st+1;
      // communication
      if (rank == mnode) ss = MPI_Wtime();
      
      MPI_Send(&rsize, 1, MPI_INT, rank-1, 3, W_COMM);
      MPI_Send(&nums[st], ed-st+1, MPI_INT, rank-1, 4, W_COMM);
      // receive sorted
      MPI_Recv(&nums[st], ed-st+1, MPI_INT, rank-1, 5, W_COMM, &suc);

      if (rank == mnode) commtime += MPI_Wtime()-ss;
    } else if ((rank%2 == 0) && (rank+1<size)) {
      int rsize;
      // communication
      if (rank == mnode) ss = MPI_Wtime();

      MPI_Recv(&rsize, 1, MPI_INT, rank+1, 3, W_COMM, &suc);
      MPI_Recv(&nums[ed+1], rsize, MPI_INT, rank+1, 4, W_COMM, &suc);

      if (rank == mnode) commtime += MPI_Wtime()-ss;
      // computation: merge
      if (rank == mnode) ss = MPI_Wtime();
      
      tmp_merge = merge(&nums[st], &nums[ed+1], ed-st+1, rsize);
      for (i=0; i<rsize+(ed-st+1); i++) nums[st+i] = tmp_merge[i];

      if (rank == mnode) cmptime += MPI_Wtime()-ss;
      // communication
      if (rank == mnode) ss = MPI_Wtime();
      MPI_Send(&nums[ed+1], rsize, MPI_INT, rank+1, 5, W_COMM);
      if (rank == mnode) commtime += MPI_Wtime()-ss;
    }
  }
  free(tmp_merge);
  
  // communication: gather all
  if (rank == mnode) ss = MPI_Wtime();
  
  if (rank+1<size) MPI_Recv(&nums[ed+1], len-ed-1, MPI_INT, rank+1, 6, W_COMM, &suc);
  if (rank>=1) MPI_Send(&nums[st], len-st, MPI_INT, rank-1, 6, W_COMM);

  if (rank == mnode) commtime += MPI_Wtime()-ss;
  return nums;
}
void writeMeasure(double tot, double cmptime, double commtime, double iotime, FILE *fmse, int size) {
  char fname[100];
  sprintf(fname, "%s_%d.txt", "adv_mse", size);
  fmse = fopen(fname, "w+");
  printf("%lf %lf %lf %lf\n", cmptime, commtime, iotime, tot);
  fprintf(fmse, "%lf %lf %lf %lf\n", cmptime, commtime, iotime, tot);
}
