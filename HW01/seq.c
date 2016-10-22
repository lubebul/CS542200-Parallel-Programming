#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define max(x,y) (x)>(y)?(x):(y)
#define min(x,y) (x)<(y)?(x):(y)
// measure time
double iotime;
int mnode;

int main (int argc, char** argv) {
  int rank, size;
  // measure time
  double ss;
  FILE *fmse; iotime = 0.0; mnode = 0; ss = 0;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == mnode) ss = MPI_Wtime();
  // dealing with I/Os
  assert(("Usage: ./HW_104062703_advanced N in-file out-file\n") && argc == 4);
  // seq I/O
  FILE *fin, *fout;
  if (rank == mnode) {
    fin = fopen(argv[2], "r");
    // read number array
    int N = atoi(argv[1]); int *nums = malloc(N*sizeof(int));
    fread(nums, sizeof(int), N, fin);
    fout = fopen(argv[3], "w+");
    fwrite(nums, sizeof(int), N, fout);
    fclose(fin); fclose(fout);
    iotime += MPI_Wtime()-ss;
    free(nums);
    char fname[100];
    sprintf(fname, "%s_%d.txt", "seq_mse", size);
    fmse = fopen(fname, "w+");
    printf("%lf\n", iotime);
    fprintf(fmse, "%lf\n", iotime);
  }
  
  MPI_Finalize();
  return 0;
}
