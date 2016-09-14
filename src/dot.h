/*----------------------------------------------------------------------------
  File    : dot.h
  Contents: dot product (cpu dispatcher)
  Author  : Kristian Loewe
----------------------------------------------------------------------------*/
#ifndef DOT_H
#define DOT_H

#ifdef __cplusplus
extern "C"
{
#endif

/*----------------------------------------------------------------------------
  Type Definitions
----------------------------------------------------------------------------*/
typedef float  (sdot_func)  (const float  *a, const float  *b, int n);
typedef double (ddot_func)  (const double *a, const double *b, int n);
typedef double (sddot_func) (const float  *a, const float  *b, int n);

/*----------------------------------------------------------------------------
  Global Variables
----------------------------------------------------------------------------*/
extern sdot_func  *sdot_ptr;
extern ddot_func  *ddot_ptr;
extern sddot_func *sddot_ptr;

/*----------------------------------------------------------------------------
  Function Prototypes
----------------------------------------------------------------------------*/
inline float  sdot         (const float  *a, const float  *b, int n);
inline double ddot         (const double *a, const double *b, int n);
inline double sddot        (const float  *a, const float  *b, int n);

extern float  sdot_select  (const float  *a, const float  *b, int n);
extern double ddot_select  (const double *a, const double *b, int n);
extern double sddot_select (const float  *a, const float  *b, int n);

extern float  sdot_avx     (const float  *a, const float  *b, int n);
extern double ddot_avx     (const double *a, const double *b, int n);
extern double sddot_avx    (const float  *a, const float  *b, int n);

extern float  sdot_sse2    (const float  *a, const float  *b, int n);
extern double ddot_sse2    (const double *a, const double *b, int n);
extern double sddot_sse2   (const float  *a, const float  *b, int n);

extern float  sdot_naive   (const float  *a, const float  *b, int n);
extern double ddot_naive   (const double *a, const double *b, int n);
extern double sddot_naive  (const float  *a, const float  *b, int n);

/*----------------------------------------------------------------------------
  Inline Functions
----------------------------------------------------------------------------*/

inline float sdot (const float *a, const float *b, int n) {
  return (*sdot_ptr)(a,b,n);
}

inline double ddot (const double *a, const double *b, int n) {
  return (*ddot_ptr)(a,b,n);
}

inline double sddot (const float *a, const float *b, int n) {
  return (*sddot_ptr)(a,b,n);
}

#ifdef __cplusplus
}
#endif

#endif  // #ifndef DOT_H
