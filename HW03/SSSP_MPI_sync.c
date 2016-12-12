#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define INF 1 << 30

typedef struct neighbor {
  int node;
  int weight;
  struct neighbor *next;
} neighbor;

void print(int *L, int src, int cur, FILE *fout);
neighbor **add(neighbor **nodes, int a, int b, int weight, int src_vtx);
void Moore(int rank, neighbor *node, int src_vtx);

int main(int argc, char** argv) {
  assert(("Usage: ./[exe] [#thread] [fin] [fout] [src_vertex]\n") && argc == 5);
  int num_vtx, num_edge, i, a, b, w;
  int src_vtx = atoi(argv[4]); src_vtx -= 1;
  // read input
  FILE *fin, *fout;
  fin = fopen(argv[2], "r"); fout = fopen(argv[3], "w+");
  fscanf(fin, "%d %d", &num_vtx, &num_edge);
  neighbor **nodes = malloc(sizeof(neighbor *)*num_vtx);
  for (i=0; i<num_vtx; i++) nodes[i] = NULL;
  for (i=0; i<num_edge; i++) {
    fscanf(fin, "%d %d %d", &a, &b, &w);
    nodes = add(nodes, a-1, b-1, w, src_vtx);
    nodes = add(nodes, b-1, a-1, w, src_vtx);
   }

  int rank, size;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double st, ed;
  if (rank == 0) st = MPI_Wtime();
  Moore(rank, nodes[rank], src_vtx);
  if (rank == 0) {
    ed = MPI_Wtime();
    printf("%d\n", ed-st);
  }
  if (rank == src_vtx) {
    MPI_Status suc;
    int *L = malloc(sizeof(int)*num_vtx);
    // collect parent L
    for (i=0; i<num_vtx; i++) {
      if (i == src_vtx) continue;
      MPI_Recv(&L[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD, &suc);
    }
    // write output
    for (i=0;i<num_vtx; i++) {
      if (i == src_vtx) fprintf(fout, "%d %d", src_vtx+1, src_vtx+1);
      else print(L, src_vtx, i, fout);
      fprintf(fout, "\n");
    }
  }
  
  MPI_Finalize();
  
  return 0;
}

neighbor **add(neighbor **nodes, int a, int b, int weight, int src_vtx) {
  neighbor *cur = malloc(sizeof(neighbor));
  cur->node = b; cur->weight = weight; cur->next = NULL;
  if (nodes[a] == NULL) nodes[a] = cur;
  else {
    neighbor *head = nodes[a];
    while(head->next != NULL) head = head->next;
    head->next = cur;
  }
  return nodes;
}

void Moore(int rank, neighbor *node, int src_vtx) {
  neighbor *head; MPI_Status suc;
  int L = -1; int D = (rank==src_vtx)?0:INF; int done = 0;  
  while (!done) {
    // send to neighbor
    head = node;
    while(head != NULL) {
      int d = D+head->weight;
      MPI_Send(&d, 1, MPI_INT, head->node, 2, MPI_COMM_WORLD);
      head = head->next;
    }
    // recv new dist
    int tdone = 1;
    head = node;
    while(head != NULL) {
      int d;
      MPI_Recv(&d, 1, MPI_INT, head->node, 2, MPI_COMM_WORLD, &suc);
      if (d < D) { D = d; L = head->node; tdone = 0; }
      head = head->next;
    }
    
    // check for termination
    MPI_Allreduce(&tdone, &done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }

  // send parent
  if (rank != src_vtx) MPI_Send(&L, 1, MPI_INT, src_vtx, 1, MPI_COMM_WORLD);
}

void print(int *L, int src, int cur, FILE *fout) {
  if (cur == src) fprintf(fout, "%d", src+1);
  else {
    print(L, src, L[cur], fout);
    fprintf(fout, " %d", cur+1);
  }
}
