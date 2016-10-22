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
  MPI_File fin, fout;
  FILE *fmse; iotime = 0.0; mnode = 0; ss = 0;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
  MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
  
  // dealing with I/Os
  assert(("Usage: ./HW_104062703_advanced N in-file out-file\n") && argc == 4);
  // seq I/O
  if (rank == mnode) {
    // read number array
    int N = atoi(argv[1]); int *nums = malloc(N*sizeof(int));
    MPI_Status suc;
    ss = MPI_Wtime();
    MPI_File_read(fin, nums, N, MPI_INT, &suc); MPI_File_write(fout, nums, N, MPI_INT, &suc);
    iotime += MPI_Wtime()-ss; 
    free(nums);
    char fname[100];
    sprintf(fname, "%s_%d.txt", "mpi_mse", size);
    fmse = fopen(fname, "w+");
    printf("%lf\n", iotime);
    fprintf(fmse, "%lf\n", iotime);
  }
  
  MPI_File_close(&fin); MPI_File_close(&fout);
  MPI_Finalize();
  return 0;
}
