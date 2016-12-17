#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define INF 1 << 30
#define UPDATE 1
#define TERMINATE 2
#define COLLECT 3

typedef struct neighbor {
  int node;
  int weight;
  struct neighbor *next;
} neighbor;

double comm, comp, sync, io;
int msg;
int *Last;
void print(int *L, int src, int cur, FILE *fout);
neighbor **add(neighbor **nodes, int a, int b, int weight, int src_vtx);
void Moore(int rank, neighbor *node, int src_vtx, int size);

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
  // io
  st = MPI_Wtime();
  fscanf(fin, "%d %d", &num_vtx, &num_edge);
  io += MPI_Wtime()-st;
  
  neighbor **nodes = malloc(sizeof(neighbor *)*num_vtx);
  for (i=0; i<num_vtx; i++) nodes[i] = NULL;
  // io
  st = MPI_Wtime();
  for (i=0; i<num_edge; i++) {
    fscanf(fin, "%d %d %d", &a, &b, &w);
    nodes = add(nodes, a-1, b-1, w, src_vtx);
    nodes = add(nodes, b-1, a-1, w, src_vtx);
  }
  io += MPI_Wtime()-st;
  Last = NULL;
  
  Moore(rank, nodes[rank], src_vtx, size);

  if (rank == src_vtx) {
    st = MPI_Wtime();
    // write output
    for (i=0;i<num_vtx; i++) {
      if (i == src_vtx) fprintf(fout, "%d %d", src_vtx+1, src_vtx+1);
      else print(Last, src_vtx, i, fout);
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

void Moore(int rank, neighbor *node, int src_vtx, int size) {
  neighbor *head; MPI_Status suc;
  int L = -1; int D = (rank==src_vtx)?0:INF; int done = 0; int color = 1; int passed = 0; int d; int once = 0; int count = 0;
  double st, t, tmp;

  // init: src node => comp & comm
  if (rank == src_vtx) {
    st = MPI_Wtime(); tmp = 0.0;
    head = node;
    while(head != NULL) {
      d = D+head->weight;
      if (head->node < rank) color = 0;
      t = MPI_Wtime();
      MPI_Send(&d, 1, MPI_INT, head->node, UPDATE, MPI_COMM_WORLD);
      msg += 1;
      tmp += MPI_Wtime()-t;
      head = head->next;
    }
    comp += MPI_Wtime()-st-tmp;
    comm += tmp;
  }
  
  while (!done) {
    st = MPI_Wtime(); tmp = 0.0;
    // comm
    t = MPI_Wtime();
    MPI_Recv(&d, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &suc);
    tmp += MPI_Wtime()-t;
    
    switch(suc.MPI_TAG) {
    case UPDATE: {
      if (d < D) {
	D = d; L = suc.MPI_SOURCE; head = node;
	while(head != NULL) {
	  d = D+head->weight;
	  if (head->node < rank) color = 0;
	  // comm
	  t = MPI_Wtime();
	  MPI_Send(&d, 1, MPI_INT, head->node, UPDATE, MPI_COMM_WORLD);
	  msg += 1;
	  tmp += MPI_Wtime()-t;
	  
	  head = head->next;
	}
      }
      if (rank == 0 && !once) {
	// comm
	t = MPI_Wtime();
	MPI_Send(&color, 1, MPI_INT, (rank+1)%size, TERMINATE, MPI_COMM_WORLD);
	msg += 1;
	tmp += MPI_Wtime()-t;
	
	passed = 1; once = 1;}
      break;}
    case TERMINATE: {
      int token = color&d;
      if (!d && !passed && rank==0 && color) {
	// comm
	t = MPI_Wtime();
	MPI_Send(&color, 1, MPI_INT, (rank+1)%size, TERMINATE, MPI_COMM_WORLD);
	msg += 1;
	tmp += MPI_Wtime()-t;

	passed = 1;}
      else {
	if (passed==2) {
	  if(rank<size-1) {
	    t = MPI_Wtime();
	    MPI_Send(&token, 1, MPI_INT, rank+1, TERMINATE, MPI_COMM_WORLD);
	    msg += 1;
	    tmp += MPI_Wtime()-t;
	  }
	  // send parent
	  if (rank != src_vtx) {
	    t = MPI_Wtime();
	    MPI_Send(&L, 1, MPI_INT, src_vtx, COLLECT, MPI_COMM_WORLD);
	    msg += 1;
	    tmp += MPI_Wtime()-t;

	    done = 1;}
	}
	if (passed!=2) {
	  t = MPI_Wtime();
	  MPI_Send(&token, 1, MPI_INT, (rank+1)%size, TERMINATE, MPI_COMM_WORLD);
	  msg += 1;
	  tmp += MPI_Wtime()-t;

	  passed++; if(!token) passed = 0;}
	color = 1;
      }
      break;}
    case COLLECT: {
      if (Last == NULL) Last = malloc(sizeof(int)*size);
      Last[suc.MPI_SOURCE] = d;
      count ++;
      if (count == size-1) done = 1;
      break;}
    }
    comp += MPI_Wtime()-st-tmp;
    comm += tmp;
  }
}

void print(int *L, int src, int cur, FILE *fout) {
  if (cur == src) fprintf(fout, "%d", src+1);
  else {
    print(L, src, L[cur], fout);
    fprintf(fout, " %d", cur+1);
  }
}
