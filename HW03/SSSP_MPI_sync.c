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

double comm, comp, sync, io;
int msg;
void print(int *L, int src, int cur, FILE *fout);
neighbor **add(neighbor **nodes, int a, int b, int weight, int src_vtx);
void Moore(int rank, neighbor *node, int src_vtx);

int main(int argc, char** argv) {
  assert(("Usage: ./[exe] [#thread] [fin] [fout] [src_vertex]\n") && argc == 5);
  int rank, size;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // measure time
  double st, ed;
  comm = comp = sync = io = 0.0; msg = 0;
  int num_vtx, num_edge, i, a, b, w;
  int src_vtx = atoi(argv[4]); src_vtx -= 1;
  // read input
  FILE *fin, *fout;
  fin = fopen(argv[2], "r"); fout = fopen(argv[3], "w+");
  st = MPI_Wtime();
  fscanf(fin, "%d %d", &num_vtx, &num_edge);
  io += MPI_Wtime()-st;
  
  neighbor **nodes = malloc(sizeof(neighbor *)*num_vtx);
  for (i=0; i<num_vtx; i++) nodes[i] = NULL;

  st = MPI_Wtime();
  for (i=0; i<num_edge; i++) {
    fscanf(fin, "%d %d %d", &a, &b, &w);
    nodes = add(nodes, a-1, b-1, w, src_vtx);
    nodes = add(nodes, b-1, a-1, w, src_vtx);
   }
  io += MPI_Wtime()-st;
  
  Moore(rank, nodes[rank], src_vtx);
  
  if (rank == src_vtx) {
    st = MPI_Wtime();
    MPI_Status suc;
    int *L = malloc(sizeof(int)*num_vtx);
    // collect parent L
    for (i=0; i<num_vtx; i++) {
      if (i == src_vtx) continue;
      MPI_Recv(&L[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD, &suc);
    }
    comm += MPI_Wtime()-st;

    st = MPI_Wtime();
    // write output
    for (i=0;i<num_vtx; i++) {
      if (i == src_vtx) fprintf(fout, "%d %d", src_vtx+1, src_vtx+1);
      else print(L, src_vtx, i, fout);
      fprintf(fout, "\n");
    }
    io += MPI_Wtime()-st;
  }
  
  MPI_Finalize();

  printf("[%d] %lf %lf %lf %lf %d\n", rank, comm, comp, sync, io, msg);
  
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
  neighbor *head; MPI_Status suc; double st, ed, t, tmp;
  int L = -1; int D = (rank==src_vtx)?0:INF; int done = 0;  
  while (!done) {
    // send to neighbor
    head = node;
    // comp & comm
    st = MPI_Wtime(); tmp = 0;
    while(head != NULL) {
      int d = D+head->weight;
      t = MPI_Wtime();
      MPI_Send(&d, 1, MPI_INT, head->node, 2, MPI_COMM_WORLD);
      msg += 1;
      tmp += MPI_Wtime()-t;
      head = head->next;
    }
    comp += MPI_Wtime()-st-tmp;
    comm += tmp;
    
    // recv new dist
    int tdone = 1;
    head = node;
    // comp & comm
    st = MPI_Wtime(); tmp = 0;
    while(head != NULL) {
      int d;
      t = MPI_Wtime();
      MPI_Recv(&d, 1, MPI_INT, head->node, 2, MPI_COMM_WORLD, &suc);
      tmp += MPI_Wtime()-t;
      if (d < D) { D = d; L = head->node; tdone = 0; }
      head = head->next;
    }
    comp += MPI_Wtime()-st-tmp;
    comm += tmp;

    st = MPI_Wtime();
    // check for termination
    MPI_Allreduce(&tdone, &done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    msg += 1;
    sync += MPI_Wtime()-st;
  }

  st = MPI_Wtime();
  // send parent
  if (rank != src_vtx) { MPI_Send(&L, 1, MPI_INT, src_vtx, 1, MPI_COMM_WORLD); msg += 1;}
  comm += MPI_Wtime()-st;
}

void print(int *L, int src, int cur, FILE *fout) {
  if (cur == src) fprintf(fout, "%d", src+1);
  else {
    print(L, src, L[cur], fout);
    fprintf(fout, " %d", cur+1);
  }
}
