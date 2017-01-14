#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BSIZE 32
#define HANDLE_ERROR(status) {if (status != cudaSuccess) { fprintf(stderr, "%s failed  at line %d \nError message: %s \n", __FILE__, __LINE__ ,cudaGetErrorString(status)); exit(EXIT_FAILURE);}}

const int INF = 10000000;

void input(char *inFileName);
void output(char *outFileName);
int ceil(int a, int b);

int init_device();
__global__ void cal(int B, int n, int **dev_Dist, int Round, int bx_st, int bx_ed, int by_st, int by_ed);
__host__ void block_FW(int n, int B);

int n, m;
int **Dist;

int main(int argc, char* argv[]) {
  input(argv[1]);
  int B = atoi(argv[3]);
  block_FW(n, B);
  output(argv[2]);
  return 0;
}

int init_device() {
  int iDeviceCount = 0;
  cudaGetDeviceCount(&iDeviceCount);
  for (int i=0; i<iDeviceCount; i++) HANDLE_ERROR(cudaSetDevice(i));
  return 0;
}
void input(char *inFileName) {
  FILE *infile = fopen(inFileName, "r");
  int size = fscanf(infile, "%d %d", &n, &m);
  // pinned Dist
  HANDLE_ERROR(cudaHostAlloc((void ***)&Dist, sizeof(int *)*n, cudaHostAllocDefault));
  for(int i=0; i<n; i++) HANDLE_ERROR(cudaHostAlloc((void **) &Dist[i], sizeof(int)*n, cudaHostAllocDefault));
  
  for (int i=0; i<n; ++i)
    for (int j=0; j<n; ++j) {
      if (i==j)	Dist[i][j] = 0; else Dist[i][j] = INF;
    }
  while (--m >= 0) {
    int a, b, v; size = fscanf(infile, "%d %d %d", &a, &b, &v);
    --a, --b; Dist[a][b] = v;
  }
}
void output(char *outFileName) {
  FILE *outfile = fopen(outFileName, "w+");
  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) {
      if (Dist[i][j] >= INF) fprintf(outfile, "INF "); else fprintf(outfile, "%d ", Dist[i][j]);
    }
    fprintf(outfile, "\n");
  }
}
int ceil(int a, int b) { return (a+b-1)/b; }

__host__ void block_FW(int n, int B) {
  // init GPU
  init_device();
  B = min(B, BSIZE); int round = ceil(n, B); dim3 grid (round, round); dim3 block (B, B, 1);
  // measure time
  cudaEvent_t st, ed, cst, ced; float t;
  cudaEventCreate(&st); cudaEventCreate(&ed); cudaEventCreate(&cst); cudaEventCreate(&ced);
  float tcmp, tcomm, tcpy;
  // malloc device memory
  int **dev_Dist; int **tmp = (int **)malloc(sizeof(int *)*n);
  cudaEventRecord(st, 0);
  HANDLE_ERROR(cudaMalloc((void ***)&dev_Dist, sizeof(int *)*n));
  for(int i=0; i<n; i++) {
    HANDLE_ERROR(cudaMalloc((void **) &tmp[i], sizeof(int)*n));
    HANDLE_ERROR(cudaMemcpyAsync(tmp[i], Dist[i], sizeof(int)*n, cudaMemcpyHostToDevice));
  }
  HANDLE_ERROR(cudaMemcpyAsync(dev_Dist, tmp, sizeof(int *)*n, cudaMemcpyHostToDevice));
  cudaEventRecord(ed, 0); cudaEventSynchronize(ed); cudaEventElapsedTime(&t, st, ed); tcpy = t/1000.0;
  
  tcomm = 0.0; cudaEventRecord(st, 0);
  for (int r = 0; r < round; r++) {
    // Phase 1
    cudaEventRecord(cst, 0);
    cal <<<grid, block>>>(B, n, dev_Dist, r, r, r, r, r);
    cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); tcmp += t/1000.0;
    // Phase 2
    cudaEventRecord(cst, 0);
    if (r > 0) {
      cal <<<grid, block>>> (B, n, dev_Dist, r, r, r, 0, r-1);
      cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r-1, r, r);
    }
    cal <<<grid, block>>> (B, n, dev_Dist, r, r, r, r+1, round-1);
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, round-1, r, r);
    cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); tcmp += t/1000.0;
    // Phase 3
    cudaEventRecord(cst, 0);
    if (r > 0) {
      cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r-1, 0, r-1);
      cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r-1, r+1, round-1);
      cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, round-1, 0, r-1);
    }
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, round-1, r+1, round-1);
    cudaEventRecord(ced, 0); cudaEventSynchronize(ced); cudaEventElapsedTime(&t, cst, ced); tcmp += t/1000.0;
  }
  cudaEventRecord(ed, 0); cudaEventSynchronize(ed); cudaEventElapsedTime(&t, st, ed); tcomm += t/1000.0-tcmp;
  // copy result
  cudaEventRecord(st, 0);
  cudaMemcpy(tmp, dev_Dist, sizeof(int *)*n, cudaMemcpyDeviceToHost);
  for(int i=0; i<n; i++) cudaMemcpy(Dist[i], tmp[i], sizeof(int)*n, cudaMemcpyDeviceToHost);
  cudaEventRecord(ed, 0); cudaEventSynchronize(ed); cudaEventElapsedTime(&t, st, ed); tcpy += t/1000.0;
  printf("%.3lf %.3lf %.3lf\n", tcmp, tcomm, tcpy);
  for (int i=0; i<n; i++) cudaFree(&dev_Dist[i]);
  cudaFree(dev_Dist);
  free(tmp);
}
__global__ void cal(int B, int n, int **dev_Dist, int Round, int bx_st, int bx_ed, int by_st, int by_ed) {
  int bx = blockIdx.x; int by = blockIdx.y;
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
