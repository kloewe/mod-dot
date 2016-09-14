/*----------------------------------------------------------------------------
  File    : dot.c
  Contents: dot product (cpu dispatcher)
  Author  : Kristian Loewe
----------------------------------------------------------------------------*/
#include "cpuinfo.h"
#include "dot.h"

/*----------------------------------------------------------------------------
  Function Prototypes
----------------------------------------------------------------------------*/
extern float  sdot  (const float  *a, const float  *b, int n);
extern double ddot  (const double *a, const double *b, int n);
extern double sddot (const float  *a, const float  *b, int n);

/*----------------------------------------------------------------------------
  Global Variables
----------------------------------------------------------------------------*/
sdot_func  *sdot_ptr  = &sdot_select;
ddot_func  *ddot_ptr  = &ddot_select;
sddot_func *sddot_ptr = &sddot_select;

/*----------------------------------------------------------------------------
  Functions
----------------------------------------------------------------------------*/

float sdot_select (const float *a, const float *b, int n) {
  if      (hasAVX())
    sdot_ptr = &sdot_avx;
  else if (hasSSE2())
    sdot_ptr = &sdot_sse2;
  else
    sdot_ptr = &sdot_naive;
  return (*sdot_ptr)(a,b,n);
}

double ddot_select (const double *a, const double *b, int n) {
  if      (hasAVX())
    ddot_ptr = &ddot_avx;
  else if (hasSSE2())
    ddot_ptr = &ddot_sse2;
  else
    ddot_ptr = &ddot_naive;
  return (*ddot_ptr)(a,b,n);
}

double sddot_select (const float *a, const float *b, int n) {
  if      (hasAVX())
    sddot_ptr = &sddot_avx;
  else if (hasSSE2())
    sddot_ptr = &sddot_sse2;
  else
    sddot_ptr = &sddot_naive;
  return (*sddot_ptr)(a,b,n);
}
