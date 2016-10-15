#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define max(x,y) (x)>(y)?(x):(y)
#define min(x,y) (x)<(y)?(x):(y)

typedef struct Message {
  int idx; int value;
} Message;

void openFiles(char* in, char* out);
void closeFiles(void);
void printNums(int len, int* nums);
void swap (int *nums, int i, int j);
int* seqOddEvenSort(int len, int *nums);
int* parOddEvenSort(int len, int *nums, int rank, int size, int chunk, MPI_Comm W_COMM);

int main (int argc, char** argv) {
  int rank, size;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_File fin, fout;
  // dealing with I/Os
  assert(("Usage: ./HW_104062703_basic N in-file out-file\n") && argc == 4);
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
  MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
  // read number array
  int N = atoi(argv[1]); int *nums = malloc(N*sizeof(int));
  MPI_Status suc;
  MPI_File_read(fin, nums, N, MPI_INT, &suc);
  
  if (N < 7) { // too less items
    printf("seq\n");
    if (rank == 0) { seqOddEvenSort(N, nums); MPI_File_write(fout, nums, N, MPI_INT, &suc);}
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fin); MPI_File_close(&fout); MPI_Finalize();
    return 0;
  }
  
  // extract working groups
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
  int chunk = (N+size-1) / size;
  if (rank < size) nums = parOddEvenSort(N, nums, rank, size, chunk, W_COMM);
  if (rank == 0) MPI_File_write(fout, nums, N, MPI_INT, &suc);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fin); MPI_File_close(&fout);
  free(nums);
  MPI_Finalize();
  return 0;
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
int* parOddEvenSort(int len, int *nums, int rank, int size, int chunk, MPI_Comm W_COMM) {
  // declare new MPI_Type MSG
  Message msg[2];
  MPI_Datatype MSG; MPI_Datatype type[2] = {MPI_INT, MPI_INT}; int blocklen[2] = {1, 1}; MPI_Aint disp[2] = {0, 1};
  MPI_Type_create_struct(2, blocklen, disp, type, &MSG); MPI_Type_commit(&MSG);
  int sorted = 0; int st = min(rank*chunk, len-1); int ed = min(st + chunk-1, len-1);
  int odd, even; if (st%2) {odd = st; even = st-1;} else {odd = ((st-1)>0)?(st-1):(st+1); even = st; }
  int i; MPI_Status suc;
  while (!sorted) {
    int tsorted = 1;
    // odd phase
    for (i=odd; i<ed; i+=2) if (nums[i] > nums[i+1]) { swap(nums, i, i+1); tsorted = 0;}
    if (chunk%2 == 0) { // chunk size is even
      if (rank>=1) {msg[0].idx = odd; msg[0].value=nums[odd]; MPI_Send(&msg[0], 1, MSG, rank-1, 0, W_COMM);}
      if (rank+1<size) {MPI_Recv(&msg[1], 1, MSG, rank+1, 0, W_COMM, &suc); nums[msg[1].idx] = msg[1].value;}
    } else { // chunk size is odd
      if ((odd<st) && (rank>=1)) {msg[0].idx = odd; msg[0].value=nums[odd]; MPI_Send(&msg[0], 1, MSG, rank-1, 0, W_COMM);}
      else if ((rank+1<size) && (st%2 == 1)) {MPI_Recv(&msg[1], 1, MSG, rank+1, 0, W_COMM, &suc); nums[msg[1].idx] = msg[1].value;}
    }
    // last element
    if (rank+1<size) { msg[0].idx = i-1; msg[0].value=nums[i-1]; MPI_Send(&msg[0], 1, MSG, rank+1, 1, W_COMM);}
    if (rank>=1) { MPI_Recv(&msg[1], 1, MSG, rank-1, 1, W_COMM, &suc); nums[msg[1].idx] = msg[1].value;}
    
    // even phase
    for (i=even; i<ed; i+=2) if (nums[i] > nums[i+1]) { swap(nums, i, i+1); tsorted = 0;}
    if (chunk%2 == 1) { // chunk size is odd
      if ((even<st) && (rank>=1)) {msg[0].idx = even; msg[0].value=nums[even]; MPI_Send(&msg[0], 1, MSG, rank-1, 2, W_COMM);} else if ((rank+1<size) && (st%2 == 0)) {MPI_Recv(&msg[1], 1, MSG, rank+1, 2, W_COMM, &suc); nums[msg[1].idx] = msg[1].value;}
    }
    // last element
    if (rank+1<size) { msg[0].idx = i-1; msg[0].value=nums[i-1]; MPI_Send(&msg[0], 1, MSG, rank+1, 3, W_COMM);}
    if (rank>=1) { MPI_Recv(&msg[1], 1, MSG, rank-1, 3, W_COMM, &suc); nums[msg[1].idx] = msg[1].value;}
    MPI_Allreduce (&tsorted, &sorted, 1, MPI_INT, MPI_MIN, W_COMM);
  }
  // gather all
  for (i=0; i<size; i++) { if(i==rank) continue; MPI_Send(&nums[st], ed-st+1, MPI_INT, i, 4, W_COMM);}
  for (i=0; i<size; i++) {
    if(i==rank) continue; int tst = min(i*chunk, len-1); int ted = min(tst+chunk-1, len-1);
    MPI_Recv(&nums[tst], ted-tst+1, MPI_INT, i, 4, W_COMM, &suc);
    }
  
  return nums;
}
