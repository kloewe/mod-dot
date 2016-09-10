/*----------------------------------------------------------------------------
  File    : dot_sse2.h
  Contents: dot product (SSE2-based implementations)
  Author  : Kristian Loewe, Christian Borgelt
----------------------------------------------------------------------------*/
#ifndef DOT_SSE2_H
#define DOT_SSE2_H

#ifndef __SSE2__
#  error "SSE2 is not enabled"
#endif

#include <emmintrin.h>

// horizontal sum variant
#ifdef HORZSUM_SSE3
#  ifdef __SSE3__
#    include <pmmintrin.h>
#  else
#    error "HORZSUM_SSE3 requires SSE3."
#  endif
#endif

/*----------------------------------------------------------------------------
  Function Prototypes
----------------------------------------------------------------------------*/
inline float  sdot_sse2  (const float  *a, const float  *b, int n);
inline double ddot_sse2  (const double *a, const double *b, int n);
inline double sddot_sse2 (const float  *a, const float  *b, int n);

/*----------------------------------------------------------------------------
  Inline Functions
----------------------------------------------------------------------------*/

// --- dot product (single precision)
inline float sdot_sse2 (const float *a, const float *b, int n)
{
  // initialize 4 sums
  __m128 s4 = _mm_setzero_ps();

  // in each iteration, add 1 product to each of the 4 sums in parallel
  for (int k = 0; k < 4*(n/4); k += 4)
    s4 = _mm_add_ps(s4, _mm_mul_ps(_mm_loadu_ps(a+k), _mm_loadu_ps(b+k)));

  // compute horizontal sum
  #ifdef HORZSUM_SSE3
  s4 = _mm_hadd_ps(s4,s4);
  s4 = _mm_hadd_ps(s4,s4);
  #else
  s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
  #endif
  float s = _mm_cvtss_f32(s4);  // extract horizontal sum from 1st elem.

  // add the remaining products
  for (int k = 4*(n/4); k < n; k++)
    s += a[k] * b[k];

  return s;
}  // sdot_sse2()

/*--------------------------------------------------------------------------*/

// --- dot product (double precision)
inline double ddot_sse2 (const double *a, const double *b, int n)
{
  // initialize 2 sums
  __m128d s2 = _mm_setzero_pd();

  // in each iteration, add 1 product to each of the 2 sums in parrallel
  for (int k = 0; k < 2*(n/2); k += 2)
    s2 = _mm_add_pd(s2, _mm_mul_pd(_mm_loadu_pd(a+k), _mm_loadu_pd(b+k)));

  // compute horizontal sum
  #ifdef HORZSUM_SSE3
  s2 = _mm_hadd_pd(s2,s2);
  #else
  s2 = _mm_add_pd(s2, _mm_shuffle_pd(s2, s2, 1));
  #endif
  double s = _mm_cvtsd_f64(s2); // extract horizontal sum from 1st elem.

  // add the remaining products
  for (int k = 2*(n/2); k < n; k++)
    s += a[k] * b[k];

  return s;
}  // ddot_sse2()

/*--------------------------------------------------------------------------*/

// --- dot product (input: single; intermediate and output: double)
inline double sddot_sse2 (const float *a, const float *b, int n)
{
  // initialize 2 sums
  __m128d s2 = _mm_setzero_pd();

  // in each iteration, add 1 product to each of the 2 sums in parrallel
  for (int k = 0; k < 2*(n/2); k += 2)
    s2 = _mm_add_pd(s2, _mm_mul_pd(_mm_cvtps_pd(_mm_loadu_ps(a+k)),
                                   _mm_cvtps_pd(_mm_loadu_ps(b+k))));

  // compute horizontal sum
  #ifdef HORZSUM_SSE3
  s2 = _mm_hadd_pd(s2,s2);
  #else
  s2 = _mm_add_pd(s2, _mm_shuffle_pd(s2, s2, 1));
  #endif
  double s = _mm_cvtsd_f64(s2); // extract horizontal sum from 1st elem.

  // add the remaining products
  for (int k = 2*(n/2); k < n; k++)
    s += a[k] * b[k];

  return s;
}  // sddot_sse2()

#endif // DOT_SSE2_H
