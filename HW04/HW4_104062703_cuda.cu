#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define min(x,y) (x)<(y)?(x):(y)
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
  cudaMalloc((void ***)&dev_Dist, sizeof(int *)*n);
  for(int i=0; i<n; i++) {
    cudaMalloc((void **) &tmp[i], sizeof(int)*n);
    cudaMemcpy(tmp[i], Dist[i], sizeof(int)*n, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(dev_Dist, tmp, sizeof(int *)*n, cudaMemcpyHostToDevice);
  // clipping
  B = min(B, n);
  int round = ceil(n, B);
  dim3 grid (round, round, 1);
  dim3 block (B, B, 1);
  for (int r = 0; r < round; ++r) {
    /* Phase 1*/
    cal <<<grid, block>>> (B, n, dev_Dist, r, r, r, 1, 1);
    cudaDeviceSynchronize();
    
    /* Phase 2*/
    cal <<<grid, block>>> (B, n, dev_Dist, r, r, 0, r, 1);
    cal <<<grid, block>>> (B, n, dev_Dist, r, r,r+1, round-r-1, 1);
    cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r, 1, r);
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, r, 1, round-r-1);
    cudaDeviceSynchronize();
    
    /* Phase 3*/
    cal <<<grid, block>>> (B, n, dev_Dist, r, 0, 0, r, r);
    cal <<<grid, block>>> (B, n, dev_Dist, r, 0, r+1, round-r-1, r);
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, 0, r, round-r-1);
    cal <<<grid, block>>> (B, n, dev_Dist, r, r+1, r+1, round-r-1, round-r-1);
    cudaDeviceSynchronize();
  }
  // copy result
  cudaMemcpy(tmp, dev_Dist, sizeof(int *)*n, cudaMemcpyDeviceToHost);
  for(int i=0; i<n; i++) cudaMemcpy(Dist[i], tmp[i], sizeof(int)*n, cudaMemcpyDeviceToHost);
}

__global__ void cal(int B, int n, int **dev_Dist, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
  int block_end_x = block_start_x+block_height;
  int block_end_y = block_start_y+block_width;

  int a = min(blockIdx.x*B+threadIdx.x, n-1);
  int b = min(blockIdx.y*B+threadIdx.y, n-1);

  if ((block_start_x <= blockIdx.x && blockIdx.x <block_end_x) && (block_start_y <= blockIdx.y && blockIdx.y <block_end_y)) {
      for (int k = Round*B; k < (Round+1)*B && k<n; k++) {
        atomicMin(&dev_Dist[a][b], dev_Dist[a][k]+dev_Dist[k][b]);
      }
    }
}
