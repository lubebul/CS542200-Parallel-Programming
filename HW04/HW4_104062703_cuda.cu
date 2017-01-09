#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BSIZE 32
#define HANDLE_ERROR(status) {if (status != cudaSuccess) { fprintf(stderr, "%s failed  at line %d \nError message: %s \n", __FILE__, __LINE__ ,cudaGetErrorString(status)); exit(EXIT_FAILURE);}}
const int INF = 10000000;

int init_device();
void input(char *inFileName);
void output(char *outFileName);
void print(int **dist, int n);

int ceil(int a, int b);
__global__ void cal(int B, int n, int **dev_Dist, int Round, int block_start_x, int block_start_y, int block_width, int block_height);
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
  // malloc Dist
  Dist = (int **) malloc(sizeof(int*)*n);
  for (int i=0; i<n; i++) Dist[i] = (int *)malloc(sizeof(int)*n);
  
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
void print(int **dist, int n) {
  printf("#########################\n");
  for (int i=0; i<n; i++) { for (int j=0; j<n; j++) printf("%d ", dist[i][j]); printf("\n"); }
  printf("#########################\n");
}
int ceil(int a, int b) { return (a+b-1)/b; }

__host__ void block_FW(int n, int B) {
  // init GPU
  init_device();
  
  // malloc device memory
  int **dev_Dist;
  int **tmp = (int **)malloc(sizeof(int *)*n);
  HANDLE_ERROR(cudaMalloc((void ***)&dev_Dist, sizeof(int *)*n));
  for(int i=0; i<n; i++) {
    HANDLE_ERROR(cudaMalloc((void **) &tmp[i], sizeof(int)*n));
    HANDLE_ERROR(cudaMemcpy(tmp[i], Dist[i], sizeof(int)*n, cudaMemcpyHostToDevice));
  }
  HANDLE_ERROR(cudaMemcpy(dev_Dist, tmp, sizeof(int *)*n, cudaMemcpyHostToDevice));
  // clipping
  B = min(B, BSIZE);
  int round = ceil(n, B);
  dim3 grid (round, round);
  dim3 block (B, B, 1);
  
  for (int r = 0; r < round; r++) {
    /* Phase 1*/
    cal <<<grid, block>>> (B, n, dev_Dist, r, r, r, r, r);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaPeekAtLastError());
    
    /* Phase 2*/
    if (r > 0) {
      cal <<<grid, block>>> (B, n, dev_Dist, r, r, r, 0, r-1);
      cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r-1, r, r);
    }
    cal <<<grid, block>>> (B, n, dev_Dist, r, r, r, r+1, round-1);
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, round-1, r, r);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaPeekAtLastError());
    
    /* Phase 3*/
    if (r > 0) {
      cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r-1, 0, r-1);
      cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r-1, r+1, round-1);
      cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, round-1, 0, r-1);
    }
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, round-1, r+1, round-1);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaPeekAtLastError());
  }
  // copy result
  cudaMemcpy(tmp, dev_Dist, sizeof(int *)*n, cudaMemcpyDeviceToHost);
  for(int i=0; i<n; i++) cudaMemcpy(Dist[i], tmp[i], sizeof(int)*n, cudaMemcpyDeviceToHost);
}

__global__ void cal(int B, int n, int **dev_Dist, int Round, int bx_st, int bx_ed, int by_st, int by_ed) {
  int x = threadIdx.x; int y = threadIdx.y;
  int a = blockIdx.x*B + x; int b = blockIdx.y*B + y;
  
  if ((bx_st*B<=a && a<min((bx_ed+1)*B,n)) && (by_st*B<=b && b<min((by_ed+1)*B,n))) {\
    #pragma unroll
    for (int k = Round*B; k < min((Round+1)*B,n); k++) {
      atomicMin(&dev_Dist[a][b], dev_Dist[a][k]+dev_Dist[k][b]);
      __syncthreads();
    }
    /*
    __shared__ int d[BSIZE][BSIZE];
    __shared__ int AK[BSIZE][BSIZE];
    __shared__ int KB[BSIZE][BSIZE];
    d[x][y] = dev_Dist[a][b];
    if (Round*B+y < n) AK[x][y] = dev_Dist[a][Round*B+y]; else AK[x][y] = INF;
    if (Round*B+x < n) KB[x][y] = dev_Dist[Round*B+x][b]; else KB[x][y] = INF;
    __syncthreads();

    int d1, d2;
    #pragma unroll
    for (int k = Round*B; k < min((Round+1)*B,n); k++) {
      if (Round == blockIdx.y) d1 = d[x][k%B]; else d1 = AK[x][k%B];
      if (Round == blockIdx.x) d2 = d[k%B][y]; else d2 = KB[k%B][y];
      d[x][y] = min(d[x][y], d1+d2);
      __syncthreads();
    }
    dev_Dist[a][b] = d[x][y];
    __syncthreads();
    */
  }
}
