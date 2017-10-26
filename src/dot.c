/*----------------------------------------------------------------------------
  File    : dot.c
  Contents: dot product (cpu dispatcher)
  Author  : Kristian Loewe
----------------------------------------------------------------------------*/
#include "dot.h"
#ifdef ARCH_IS_X86_64
#include "cpuinfo.h"
#endif

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
  dot_set_impl(DOT_AUTO);
  return (*sdot_ptr)(a,b,n);
}

double ddot_select (const double *a, const double *b, int n) {
  dot_set_impl(DOT_AUTO);
  return (*ddot_ptr)(a,b,n);
}

double sddot_select (const float *a, const float *b, int n) {
  dot_set_impl(DOT_AUTO);
  return (*sddot_ptr)(a,b,n);
}

dot_flags dot_set_impl (dot_flags impl) {

  #ifndef ARCH_IS_X86_64
  impl = DOT_NAIVE;
  #endif

  switch (impl) {
    case DOT_AUTO :
    #ifndef DOT_NOAVX512
     #ifndef DOT_NOFMA
    case DOT_AVX512FMA :
      if (hasAVX512f() && hasFMA3()) {
        sdot_ptr  = &sdot_avx512fma;
        ddot_ptr  = &ddot_avx512fma;
        sddot_ptr = &sddot_avx512fma;
        return DOT_AVX512FMA;
      }
     #endif
    case DOT_AVX512 :
      if (hasAVX512f()) {
        sdot_ptr  = &sdot_avx512;
        ddot_ptr  = &ddot_avx512;
        sddot_ptr = &sddot_avx512;
        return DOT_AVX512;
      }
    #endif
    #ifndef DOT_NOFMA
    // Note: The AVX-FMA implementations seem to be slower than the AVX
    // implementations and are thus only used if explicitly requested.
    case DOT_AVXFMA :
      if ((impl == DOT_AVXFMA) && hasAVX() && hasFMA3()) {
        sdot_ptr  = &sdot_avxfma;
        ddot_ptr  = &ddot_avxfma;
        sddot_ptr = &sddot_avxfma;
        return DOT_AVXFMA;
      }
    #endif
    case DOT_AVX :
      if (hasAVX()) {
        sdot_ptr  = &sdot_avx;
        ddot_ptr  = &ddot_avx;
        sddot_ptr = &sddot_avx;
        return DOT_AVX;
      }
    case DOT_SSE2 :
      if (hasSSE2()) {
        sdot_ptr  = &sdot_sse2;
        ddot_ptr  = &ddot_sse2;
        sddot_ptr = &sddot_sse2;
        return DOT_SSE2;
      }
    case DOT_NAIVE :
      sdot_ptr  = &sdot_naive;
      ddot_ptr  = &ddot_naive;
      sddot_ptr = &sddot_naive;
      return DOT_NAIVE;
    default :
      return dot_set_impl(DOT_AUTO);
  }
}  // dot_set_impl()
