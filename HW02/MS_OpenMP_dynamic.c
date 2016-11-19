#include <X11/Xlib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

#define max(x,y) (x)>(y)?(x):(y)
#define min(x,y) (x)<(y)?(x):(y)
#define MAX 100000

typedef struct complextype {
  double real, imag;
} Cmpl;

void parallel(double lreal, double rreal, double dimag, double uimag, int width, int height, int Xflag, int size);

int main(int argc, char** argv) {
  // sanity check: correct arguments
  assert(("Usage: ./{exe} N in-file out-file\n") && argc == 9);
  int num_thread = atoi(argv[1]);
  double lreal = strtod(argv[2], NULL); double rreal = strtod(argv[3], NULL);
  double dimag = strtod(argv[4], NULL); double uimag = strtod(argv[5], NULL);
  int width = atoi(argv[6]); int height = atoi(argv[7]);
  int Xflag = 1-strcmp("enable", argv[8]);
  omp_set_num_threads(num_thread);
  clock_t st, ed;
  st = clock();
  parallel(lreal, rreal, dimag, uimag, width, height, Xflag, num_thread);
  ed = clock();
  printf("%d %d %lf\n", 1, num_thread, (double)(ed-st)/CLOCKS_PER_SEC);
  return 0;
}

void parallel(double lreal, double rreal, double dimag, double uimag, int width, int height, int Xflag, int size) {
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
  
  
  int i, j;
  double xscale = (rreal-lreal)/(double)width;
  double yscale = (uimag-dimag)/(double)height;
  int color[width][height];
  
#pragma omp parallel for schedule(dynamic) num_threads(size) shared(color) collapse(2)
  for (i=0; i<width; i++) {    
    for (j=0; j<height; j++) {
      int repeats; double lengthsq, tmp;
      Cmpl *z = (Cmpl *) malloc(sizeof(Cmpl));
      Cmpl *c = (Cmpl *) malloc(sizeof(Cmpl));
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
      color[i][j] = repeats;
    }
  }
  if (Xflag) {
    for (i=0; i<width; i++)
      for (j=0; j<height; j++) {
	XSetForeground (display, gc,  1024*1024*(color[i][j]%256));
	XDrawPoint (display, window, gc, i, j);
      }
    XFlush(display);
    sleep(5);
  }
}
