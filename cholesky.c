#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cholesky.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

// Unpivoted right looking Cholesky 
void cholesky1(int N, double *A)
{
  for (int i = 0; i < N; i++) {
    // Set diagonal and scale subcolumn
    double aii = A[i+i*N];
    //    printf("aii: %f\n",aii);
    double gii = sqrt(aii);
    //    printf("gii: %f\n",gii);
    int r;
    for (r = i; r < N; r++) {
      A[r+i*N] /= gii;
    }
    // Update trailing submatrix
    if (i < N-1) {
      int c;
      for (c = i+1; c < N; c++) {
	int r;
	for (r = i+1; r < N; r++) {
	  A[r+c*N] -= A[r+i*N]*A[c+i*N];
	}
      }
    }
  }
  for (int i = 0; i < N; i++) {
    int j;
    for (j = i+1; j < N; j++) {
      A[i+N*j] = 0.0;
    }
  }
}

// Pivoted right-looking Cholesky
void cholesky2(int N, double *A, int* piv, double tol, int* rank)
{
  int i, p, j, c, r;
  int tempint;
  double tempdouble, gii;
  double* temparr = (double*) malloc(N*sizeof(double));
  for (i = 0; i < N; i++)
    piv[i] = i;
  i = 0;
  while (1) {
    // printf("i: %d\n",i);
    int pivot = i;
    double dmax = A[i+i*N];
    for (p = i+1; p < N; p++) {
      if (dmax < A[p+p*N]) {
	dmax = A[p+p*N];
	pivot = p;
      }
    }
    // printf("Max: %f\n",dmax);
    // printf("Pivot: %d\n",pivot);
    // check exit conditions
    if (dmax < tol || dmax < 0.0 || i >= N)
      break;
    
    // swap pivot
    tempint = piv[i];
    piv[i] = pivot;
    piv[pivot] = tempint;
    
    // swap pivot column
    memcpy(temparr, &A[i*N], N*sizeof(double));
    memcpy(&A[i*N], &A[pivot*N], N*sizeof(double));
    memcpy(&A[pivot*N], temparr, N*sizeof(double));
    
    // swap pivot row
    for (j = 0; j < N; j++) {
      tempdouble = A[i+j*N];
      A[i+j*N] = A[pivot+j*N];
      A[pivot+j*N] = tempdouble;
    }
    
    // scale by diagonal pivot
    gii = sqrt(dmax);
    //    printf("gii: %f\n",gii);
    for (r = i; r < N; r++) {
      A[r+i*N] /= gii;
    }
    // Update trailing submatrix
    if (i < N-1) {
      for (c = i+1; c < N; c++) {
	for (r = i+1; r < N; r++) {
	  A[r+c*N] -= A[r+i*N]*A[c+i*N];
	}
      }
    }
    i++;
  }
  *rank = i--;
  for (i = 0; i < N; i++) {
    int j;
    for (j = i+1; j < N; j++) {
      A[i+N*j] = 0.0;
    }
  }
  for (int i = 0; i < N; i++)
    for (int j = *rank; j < N; j++)
      A[i+j*N] = 0;
}

// Unpivoted left-looking Cholesky (doptf2, aka Gaxpy Cholesky)
void cholesky3(int N, double *A)
{
}

// Pivoted left-looking *lazy* Cholesky (Psi4)
void cholesky5(int N, double *A)
{
}
// Pivoted left-looking *lazy* Cholesky (Gaxpy)
void cholesky6(int N, double *A)
{
}

// Pivoted left-looking *lazy* Cholesky (Harbrecht et. al.)
void cholesky7(int N, double *A, double* G, int* piv, double tol, int* rank)
{
  int i, p, j, c, r;
  int tempint;
  double tempdouble, gii;
  double* diag = (double*) malloc(N*sizeof(double));
  printf("Matrix diagonal\n");
  for (i = 0; i < N; i++) {
    piv[i] = i;
    // compute diagonal
    diag[i] = A[i+i*N];
    printf("A[%d,%d]=%f\n",i,i,diag[i]);
  }
  // compute diagonal

  //  printf("Begin while loop\n");
  i = 0;
  while (1) {
    // printf("i: %d\n",i);
    int pivot = i;
    double dmax = diag[piv[i]];
    for (p = i+1; p < N; p++) {
      if (dmax < diag[piv[p]]) {
	dmax = diag[piv[p]];
	pivot = p;
      }
    }
    // printf("Max: %f\n",dmax);
    // printf("Pivot: %d\n",pivot);
    // check exit conditions
    if (dmax < tol || dmax < 0.0 || i >= N)
      break;
    
    // swap pivot
    tempint = piv[i];
    piv[i] = piv[pivot];
    piv[pivot] = tempint;
    
    // scale by diagonal pivot
    gii = sqrt(dmax);
    printf("i: %d, piv[i]: %d, gii: %1.2e; ",i,piv[i],gii);
    G[piv[i]+i*N] = gii;
    //    printf("Compute diagonal\n");
    // compute column of original matrix
    for (r = i+1; r < N; r++) {
      G[piv[r]+i*N] = A[piv[r]+piv[i]*N];
    }

    for (int c=0; c<N; c++)
      printf("G[%d,%d]=%1.2e\n", c,i,G[c+i*N]);
    
    //    printf("Compute column\n");
    // compute column of cholesky factor
    for (r = i+1; r < N; r++) {
      for (c = 0; c < i; c++) {
	G[piv[r]+i*N] -= G[piv[i]+c*N]*G[piv[r]+c*N];
      }
      printf("G[%d,%d]*G[%d,%d]: %1.2e; ",piv[i],i,piv[r],i,G[piv[r]+i*N]);
      G[piv[r]+i*N] /= gii;
      printf("G[%d,%d]: %1.2e; ",piv[r],i,G[piv[r]+i*N]);
      // update Schur complement diagonal
      diag[piv[r]] -= G[piv[r]+i*N]*G[piv[r]+i*N];
    }
    i++;
    printf("\n");
  }
  *rank = i--;
}

// Computes the trace norm error of the Cholesky factorization
void cholError(int N, double* A, double* G, double* error)
{
  int k;
  for (k = 0; k < N; k++) {
    double dotprod = 0;
    int imax = min(k + 1, N);
    int i;
    for (i = 0; i < imax; i++) {
      double prod = G[k + i * N] * G[k + i * N];
      dotprod += prod;
    }
    double Akk = A[k + k * N];
    (*error) += abs(Akk - dotprod);
  }
}

// Computes the trace norm error of the Cholesky factorization
void cholError2(int N, double* A, double* G, double* error)
{
  int k;
  for (k = 0; k < N; k++) {
    double dotprod = 0;
    int i;
    for (i = 0; i < N; i++) {
      double prod = G[k + i * N] * G[k + i * N];
      dotprod += prod;
    }
    double Akk = A[k + k * N];
    (*error) += abs(Akk - dotprod);
  }
}

// Computes the trace norm error of a pivoted Cholesky factor
void cholErrorPiv(int N, double* A, double* G, int* piv, double* error)
{
  int k;
  for (k = 0; k < N; k++) {
    double dotprod = 0;
    int imax = min(k + 1, N);
    int i;
    for (i = 0; i < imax; i++) {
      double prod = G[k + i * N] * G[k + i * N];
      dotprod += prod;
    }
    double Akk = A[piv[k] + piv[k] * N];
    (*error) += abs(Akk - dotprod);
    //    printf("%f\n",error);
  }
}

// Prints the matrix
void printMatrix(int N, double* A)
{
  int i;
  for (i = 0; i < N; i++) {
    int j;
    for (j = 0; j < N; j++) {
      printf("%2.1f, ", A[i+j*N]);
    }
    printf("\n");
  }
}

// Prints the matrix
void printDiag(int N, double* A)
{
  int i;
  for (i = 0; i < N; i++) {
    printf("%f\n", A[i+i*N]);
  }
}

// Evaluates the matrix from the Cholesky factor
void cholmatmul(int N, double* G, double* A, int rank)
{
  int i, j, k;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < rank; k++)
	A[i+j*N] += G[i+k*N]*G[j+k*N];
}
