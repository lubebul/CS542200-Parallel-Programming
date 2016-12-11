#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

#define INF 1 << 30

typedef struct thread_param_t {
  int tid;
  int a;
} thread_param_t;

int **G, *D, *V, *L, num_thread, num_vtx, min, cur;
pthread_mutex_t mutex;
void *relax(void *params);
void *selectMin(void *params);
void print(int *Last, int src, int cur, FILE *fout);

int main(int argc, char** argv) {
  assert(("Usage: ./[exe] [#thread] [fin] [fout] [src_vertex]\n") && argc == 5);
  int num_edge, i, j, k, a, b;
  num_thread = atoi(argv[1]);
  int src_vtx = atoi(argv[4]); src_vtx -= 1;
  // read input
  FILE *fin, *fout;
  fin = fopen(argv[2], "r"); fout = fopen(argv[3], "w+");
  fscanf(fin, "%d %d", &num_vtx, &num_edge);
  G = malloc(sizeof(int *)*num_vtx);
  for (i=0; i<num_vtx; i++) G[i] = malloc(sizeof(int)*num_vtx);
  for (i=0; i<num_vtx; i++) for (j=0; j<num_vtx; j++) G[i][j]=INF;
  D = malloc(sizeof(int)*num_vtx);
  V = malloc(sizeof(int)*num_vtx);
  L = malloc(sizeof(int)*num_vtx);
  for (i=0; i<num_edge; i++) {
    fscanf(fin, "%d %d", &a, &b);
    fscanf(fin, "%d", &G[a-1][b-1]);
    G[b-1][a-1]=G[a-1][b-1];
  }
  
  // init
  for (i=0; i<num_vtx; i++) {
    D[i] = INF; L[i] = -1; V[i] = 0;
  } D[src_vtx] = 0; L[src_vtx] = src_vtx;
  pthread_t threads[num_thread];
  thread_param_t *params = malloc(sizeof(thread_param_t)*num_thread);
  pthread_mutex_init (&mutex, NULL);
  // Dijkstra
  for (k=0; k<num_vtx; k++) {
    min = INF; cur = -1;
    for(i=0; i<num_thread; i++){
      params[i].tid = i; params[i].a = -1;
      pthread_create(&threads[i], NULL, selectMin, (void *)&params[i]);
    }
    for(i=0; i<num_thread; i++) pthread_join(threads[i], NULL);
    
    if (cur == -1) break;
    V[cur] = 1;
 
    for(i=0; i<num_thread; i++){
      params[i].tid = i; params[i].a = cur;
      pthread_create(&threads[i], NULL, relax, (void *)&params[i]);
    }
    for(i=0; i<num_thread; i++) pthread_join(threads[i], NULL);
  }
  pthread_mutex_destroy(&mutex);

  // write output
  for (i=0;i<num_vtx; i++) {
    if (i == src_vtx) fprintf(fout, "%d %d", src_vtx+1, src_vtx+1);
    else print(L, src_vtx, i, fout);
    fprintf(fout, "\n");
  }
  
  return 0;
}

void *relax(void *params) {
  thread_param_t *param = (thread_param_t *)params;
  int i;
  for (i=param->tid; i<num_vtx; i+=num_thread) {
    if (!V[i] && (D[param->a]+G[param->a][i] < D[i])) {
	D[i] = D[param->a] + G[param->a][i]; L[i] = param->a;
      }
  }
  return NULL;
}
void *selectMin(void *params) {
  thread_param_t *param = (thread_param_t *)params;
  int i;
  for (i=param->tid; i<num_vtx; i+=num_thread) {
    if (!V[i] && D[i] < min) {
      pthread_mutex_lock(&mutex);
      if (!V[i] && D[i] < min) {
	cur = i; min = D[i];
      }
      pthread_mutex_unlock(&mutex);
    }
  }
  return NULL;
}

void print(int *Last, int src, int cur, FILE *fout) {
  if (cur == src) fprintf(fout, "%d", src+1);
  else {
    print(Last, src, Last[cur], fout);
    fprintf(fout, " %d", cur+1);
  }
}
