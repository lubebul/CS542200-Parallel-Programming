#include <X11/Xlib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>

#define max(x,y) (x)>(y)?(x):(y)
#define min(x,y) (x)<(y)?(x):(y)
#define MAX 100000
#define NEXT 0
#define RECV 1
#define TERMINATE 2

typedef struct complextype {
  double real, imag;
} Cmpl;

void seq(double lreal, double rreal, double dimag, double uimag, int width, int height, int Xflag);
void master(double lreal, double rreal, double dimag, double uimag, int width, int height, int Xflag, int size);
void slave(double lreal, double rreal, double dimag, double uimag, int width, int height, int size, int rank);

int main(int argc, char** argv) {
  // sanity check: correct arguments
  assert(("Usage: ./{exe} N in-file out-file\n") && argc == 9);
  int num_thread = atoi(argv[1]);
  double lreal = strtod(argv[2], NULL); double rreal = strtod(argv[3], NULL);
  double dimag = strtod(argv[4], NULL); double uimag = strtod(argv[5], NULL);
  double st;
  int width = atoi(argv[6]); int height = atoi(argv[7]);
  int Xflag = 1-strcmp("enable", argv[8]);
  // MPI routine
  int rank, size;
  MPI_Init (&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD, &size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) st = MPI_Wtime();
  if (size == 1) seq(lreal, rreal, dimag, uimag, width, height, Xflag);
  else {
    if (rank == 0) master(lreal, rreal, dimag, uimag, width, height, Xflag, size);
    else slave(lreal, rreal, dimag, uimag, width, height, size, rank);
  }
  
  if (rank == 0) {
    printf("%d %d %lf\n", size, num_thread, MPI_Wtime()-st);
  }

  MPI_Finalize();
  return 0;
}

void seq(double lreal, double rreal, double dimag, double uimag, int width, int height, int Xflag) {
  Display *display; Window window; int screen; GC gc; XGCValues values; long valuemask=0;
  if (Xflag) {
    display = XOpenDisplay(NULL);
    screen = DefaultScreen(display);
    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, width, height, 0, BlackPixel(display, screen), WhitePixel(display, screen));
    gc = XCreateGC(display, window, valuemask, &values);
    XSetForeground (display, gc, BlackPixel (display, screen));
    XSetBackground(display, gc, 0X0000FF00);
    XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
    XMapWindow(display, window);
    XSync(display, 0);
  }
  
  int repeats; double lengthsq, tmp;
  double xscale = (rreal-lreal)/(double)width;
  double yscale = (uimag-dimag)/(double)height;
  Cmpl *z = (Cmpl *) malloc(sizeof(Cmpl)); Cmpl *c = (Cmpl *) malloc(sizeof(Cmpl));
  int i, j;
  for (i=0; i<width; i++) {
    for (j=0; j<height; j++) {
      z->real = 0.0; z->imag = 0.0;
      c->real = ((double) i*xscale) + lreal;
      c->imag = ((double) j*yscale) + dimag;
      repeats = 0; lengthsq = 0.0;
      while (repeats < MAX && lengthsq < 4.0) {
	tmp = z->real*z->real - z->imag*z->imag + c->real;
	z->imag = 2*z->imag*z->real + c->imag;
	z->real = tmp;
	lengthsq = z->real*z->real + z->imag*z->imag;
	repeats++;
      }
      if (Xflag) {
	XSetForeground (display, gc,  1024*1024*(repeats%256));
	XDrawPoint (display, window, gc, i, j);
      }
    }
  }
  if (Xflag) { XFlush(display); sleep(5); }
}

void master(double lreal, double rreal, double dimag, double uimag, int width, int height, int Xflag, int size) {
  Display *display; Window window; int screen; GC gc; XGCValues values; long valuemask=0;
  if (Xflag) {
    display = XOpenDisplay(NULL);
    screen = DefaultScreen(display);
    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, width, height, 0, BlackPixel(display, screen), WhitePixel(display, screen));
    gc = XCreateGC(display, window, valuemask, &values);
    XSetForeground (display, gc, BlackPixel (display, screen));
    XSetBackground(display, gc, 0X0000FF00);
    XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
    XMapWindow(display, window);
    XSync(display, 0);
  }
  
  MPI_Status suc; MPI_Request req;
  int i, j; int *color = (int *) malloc(sizeof(int)*(height+1));
  int row, count; row = count = 0;
  for (i=1; i<size; i++) {
    MPI_Isend(&row, 1, MPI_INT, i, NEXT, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &suc);
    row++; count++;
  }

  do {
    MPI_Irecv(color, height+1, MPI_INT, MPI_ANY_SOURCE, RECV, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &suc);
    int slave_rank = suc.MPI_SOURCE;
    count--;
    if (row < width) {
      MPI_Isend(&row, 1, MPI_INT, slave_rank, NEXT, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &suc);
      row++; count++;
    } else {
      MPI_Isend(&row, 1, MPI_INT, slave_rank, TERMINATE, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &suc);
    }
    for (j=0; j<height; j++)
      if (Xflag) {
	XSetForeground (display, gc,  1024*1024*(color[j]%256));
	XDrawPoint (display, window, gc, color[height], j);
      }
  } while (count > 0);
  
  if (Xflag) { XFlush(display); sleep(5); }
}

void slave(double lreal, double rreal, double dimag, double uimag, int width, int height, int size, int rank) {
  MPI_Status suc; MPI_Request req;
  int j; int repeats; double lengthsq, tmp;
  double xscale = (rreal-lreal)/(double)width;
  double yscale = (uimag-dimag)/(double)height;
  Cmpl *z = (Cmpl *) malloc(sizeof(Cmpl)); Cmpl *c = (Cmpl *) malloc(sizeof(Cmpl));
  int color[height+1] ;
  int row;
  int count = 0;
  double time = 0.0;
  
  MPI_Irecv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &req);
  MPI_Wait(&req, &suc);
  while(suc.MPI_TAG == NEXT) {
    count += 1;
    for (j=0; j<height; j++) {
      double st = MPI_Wtime();
      z->real = 0.0; z->imag = 0.0;
      c->real = ((double) row*xscale) + lreal;
      c->imag = ((double) j*yscale) + dimag;
      repeats = 0; lengthsq = 0.0;
      while (repeats < MAX && lengthsq < 4.0) {
	tmp = z->real*z->real - z->imag*z->imag + c->real;
	z->imag = 2*z->imag*z->real + c->imag;
	z->real = tmp;
	lengthsq = z->real*z->real + z->imag*z->imag;
	repeats++;
      }
      color[j] = repeats;
      time += MPI_Wtime()-st;
    }
    color[height] = row;
    MPI_Isend(&color[0], height+1, MPI_INT, 0, RECV, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &suc);

    // receive next command
    MPI_Irecv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &suc);
  }

  printf("[%d] %d %lf\n", rank, count*height, time);
}
