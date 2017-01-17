#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BSIZE 32
#define SMAX 16
#define HANDLE_ERROR(status) {if (status != cudaSuccess) { fprintf(stderr, "%s failed  at line %d \nError message: %s \n", __FILE__, __LINE__ ,cudaGetErrorString(status)); exit(EXIT_FAILURE);}}

const int INF = 10000000;

void input(char *inFileName);
void output(char *outFileName);
void print(int **dist, int n);
int ceil(int a, int b);

int init_device();
__global__ void cal(int B, int n, int bst, int **dev_Dist, int Round, int bx_st, int bx_ed, int by_st, int by_ed);
__host__ void block_FW(int **Dist, int gid, int size, int n, int B);
void sync(int **Dist, int **dev_Dist, int **tmp, int n, int st, int ed, int B, int gid, int size);

int n, m;
int **Dist;

int main(int argc, char* argv[]) {
  int gid, size;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &gid);
  input(argv[1]);
  int B = atoi(argv[3]);
  block_FW(Dist, gid, size, n, B);
  if (gid == 1) output(argv[2]);
  MPI_Finalize();

  return 0;
}

int init_device() {
  int iDeviceCount = 0; cudaGetDeviceCount(&iDeviceCount);
  for (int i=0; i<iDeviceCount; i++) HANDLE_ERROR(cudaSetDevice(i));
  return iDeviceCount;
}
void input(char *inFileName) {
  FILE *infile = fopen(inFileName, "r");
  int size = fscanf(infile, "%d %d", &n, &m);
  // pinned Dist
  HANDLE_ERROR(cudaHostAlloc((void ***)&Dist, sizeof(int *)*n, cudaHostAllocDefault));
  for(int i=0; i<n; i++) HANDLE_ERROR(cudaHostAlloc((void **) &Dist[i], sizeof(int)*n, cudaHostAllocDefault));
  
  for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) { if (i==j)	Dist[i][j] = 0; else Dist[i][j] = INF; }
  while (--m >= 0) {
    int a, b, v; size = fscanf(infile, "%d %d %d", &a, &b, &v);
    --a, --b; Dist[a][b] = v;
  }
}
void output(char *outFileName) {
  FILE *outfile = fopen(outFileName, "w+");
  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) { if (Dist[i][j] >= INF) fprintf(outfile, "INF "); else fprintf(outfile, "%d ", Dist[i][j]); }
    fprintf(outfile, "\n");
  }
}
void print(int **dist, int n) {
  printf("#########################\n");
  for (int i=0; i<n; i++) { for (int j=0; j<n; j++) printf("%d ", dist[i][j]); printf("\n"); }
  printf("#########################\n");
}
int ceil(int a, int b) { return (a+b-1)/b; }
void sync(int **Dist, int **dev_Dist, int **tmp, int n, int st, int ed, int B, int gid, int size) {
  MPI_Request req1, req2; MPI_Status suc1, suc2;
  for(int i=0; i<n; i++) {
    if (i < st || i>=ed) {
      MPI_Irecv(Dist[i], n, MPI_INT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &req1); MPI_Wait(&req1, &suc1);
      HANDLE_ERROR(cudaMemcpy(tmp[i], Dist[i], sizeof(int)*n, cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(&dev_Dist[i], &tmp[i], sizeof(int *), cudaMemcpyHostToDevice));
    }
    else {
      HANDLE_ERROR(cudaMemcpy(&tmp[i], &dev_Dist[i], sizeof(int *), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(Dist[i], tmp[i], sizeof(int)*n, cudaMemcpyDeviceToHost));
      for (int j=0; j<size; j++) if (j != gid) {
	  MPI_Isend(Dist[i], n, MPI_INT, j, i, MPI_COMM_WORLD, &req2); MPI_Wait(&req2, &suc2);
	}
    }
  }
}

__host__ void block_FW(int **Dist, int gid, int size, int n, int B) {
  // init GPU
  int gnum = init_device();
  gnum = size;
  printf("%d\n", gid);
  cudaSetDevice(gid);  
  cudaStream_t stream[SMAX]; for(int i=0; i<SMAX; i++) HANDLE_ERROR(cudaStreamCreate(&stream[i]));
  // measure time
  cudaEvent_t tst, ted, cst, ced; float t;
  cudaEventCreate(&tst); cudaEventCreate(&ted); cudaEventCreate(&cst); cudaEventCreate(&ced);
  float cmp, comm, cpy, tcpy; cmp = comm = cpy = tcpy = 0.0;
  // malloc device memory
  int **dev_Dist; int **tmp = (int **)malloc(sizeof(int *)*n);
  cudaEventRecord(tst, 0);
  HANDLE_ERROR(cudaMalloc((void ***)&dev_Dist, sizeof(int *)*n));
  for(int i=0; i<n; i++) {
    HANDLE_ERROR(cudaMalloc((void **) &tmp[i], sizeof(int)*n));
    HANDLE_ERROR(cudaMemcpyAsync(tmp[i], Dist[i], sizeof(int)*n, cudaMemcpyHostToDevice, stream[i%SMAX]));
  }
  HANDLE_ERROR(cudaMemcpyAsync(dev_Dist, tmp, sizeof(int *)*n, cudaMemcpyHostToDevice));
  cudaEventRecord(ted, 0); cudaEventSynchronize(ted); cudaEventElapsedTime(&t, tst, ted); cpy = t/1000.0;
  
  B = min(B, BSIZE); int round = ceil(n, B);
  int gsize = ceil(round, gnum); int psize = gsize*B;
  int bst = gid*gsize; int st = psize*gid; int ed = min(psize*(gid+1), n);
  dim3 grid (gsize, round); dim3 block (B, B, 1);
  
  cudaEventRecord(tst, 0);
  for (int r = 0; r < round; r++) {
    // Phase 1
    cudaEventRecord(cst, 0);
    cal <<<grid, block, 0, stream[0]>>>(B, n, bst, dev_Dist, r, r, r, r, r);
    cudaDeviceSynchronize(); cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); cmp += t/1000.0;
    cudaEventRecord(cst, 0);
    sync(Dist, dev_Dist, tmp, n, st, ed, B, gid, size);
    cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); tcpy += t/1000.0;
    // Phase 2
    cudaEventRecord(cst, 0);
    if (r > 0) {
      cal <<<grid, block, 0, stream[0]>>> (B, n, bst, dev_Dist, r, r, r, 0, r-1);
      cal <<<grid, block, 0, stream[1]>>> (B, n, bst, dev_Dist, r, 0, r-1, r, r);
    }
    cal <<<grid, block, 0, stream[2]>>> (B, n, bst, dev_Dist, r, r, r, r+1, round-1);
    cal <<<grid, block, 0, stream[3]>>> (B, n, bst, dev_Dist, r, r+1, round-1, r, r);
    cudaDeviceSynchronize(); cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced);
    cudaEventRecord(cst, 0);
    sync(Dist, dev_Dist, tmp, n, st, ed, B, gid, size);
    cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); tcpy += t/1000.0;
    // Phase 3
    cudaEventRecord(cst, 0);
    if (r > 0) {
      cal <<<grid, block, 0, stream[0]>>> (B, n, bst, dev_Dist, r, 0, r-1, 0, r-1);
      cal <<<grid, block, 0, stream[1]>>> (B, n, bst, dev_Dist, r, 0, r-1, r+1, round-1);
      cal <<<grid, block, 0, stream[2]>>> (B, n, bst, dev_Dist, r, r+1, round-1, 0, r-1);
    }
    cal <<<grid, block, 0, stream[3]>>> (B, n, bst, dev_Dist, r, r+1, round-1, r+1, round-1);
    cudaDeviceSynchronize(); cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); cmp += t/1000.0;
    cudaEventRecord(cst, 0);
    sync(Dist, dev_Dist, tmp, n, st, ed, B, gid, size);
    cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); tcpy += t/1000.0;
  }
  cudaEventRecord(ted, 0); cudaEventSynchronize(ted); cudaEventElapsedTime(&t, tst, ted); comm += t/1000.0-cmp-tcpy; cpy += tcpy;
  printf("%.3f %.3f %.3f\n", cmp, comm, cpy);
  for (int i=0; i<n; i++) cudaFree(&dev_Dist[i]); cudaFree(dev_Dist); free(tmp);
}
__global__ void cal(int B, int n, int bst, int **dev_Dist, int Round, int bx_st, int bx_ed, int by_st, int by_ed) {
  int bx = bst+blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int a = bx*B+tx; int b = by*B+ty;
  // not in range
  if (!(bx_st*B<=a && a<min((bx_ed+1)*B,n)) || !(by_st*B<=b && b<min((by_ed+1)*B,n))) return;
  // global -> shared
  __shared__ int ab[BSIZE][BSIZE], ak[BSIZE][BSIZE], kb[BSIZE][BSIZE];
  ab[tx][ty] = dev_Dist[a][b];
  if (Round != by) for (int k = Round*B; k < min((Round+1)*B,n); k++) ak[tx][k%B] = dev_Dist[a][k];
  if (Round != bx) for (int k = Round*B; k < min((Round+1)*B,n); k++) kb[k%B][ty] = dev_Dist[k][b];
  __syncthreads();
  int d1, d2;
#pragma unroll
  for (int k = Round*B; k < min((Round+1)*B,n); k++) {
    if (Round == by) d1 = ab[tx][k%B]; else d1 = ak[tx][k%B];
    if (Round == bx) d2 = ab[k%B][ty]; else d2 = kb[k%B][ty];
    atomicMin(&ab[tx][ty], d1+d2);
    __syncthreads();
  }
  // shared -> global
  dev_Dist[a][b] = ab[tx][ty];
  __syncthreads();
}
