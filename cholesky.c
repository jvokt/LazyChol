#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cholesky.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

// Unpivoted right looking Cholesky 
void ur_cholesky(int N, double *A)
{
  // Begin main iteration
  for (int i = 0; i < N; i++) {
    // Compute diagonal entry of Cholesky factor
    double aii = A[i+i*N];
    double gii = sqrt(aii);

    // Scale subcolumn of the Cholesky factor
    for (int r = i; r < N; r++) {
      A[r+i*N] /= gii;
    }

    // Update trailing submatrix
    if (i < N-1) {
      for (int c = i+1; c < N; c++) {
	for (int r = i+1; r < N; r++) {
	  A[r+c*N] -= A[r+i*N]*A[c+i*N];
	}
      }
    }
  }

  // Zero upper triangle
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      A[i+N*j] = 0.0;
    }
  }
}

// Pivoted right-looking Cholesky
void pr_cholesky(int N, double *A, int* piv, double tol, int* rank)
{
  // Temporary array for swaping columns
  double* temparr = (double*) malloc(N*sizeof(double));

  // Initialize pivot vector
  for (int i = 0; i < N; i++)
    piv[i] = i;
  
  // Begin main iteration
  int i = 0;
  while (i < N) {
    // Determine next pivot index
    int pivot = i;
    double dmax = A[i+i*N];
    for (int p = i+1; p < N; p++) {
      if (dmax < A[p+p*N]) {
	dmax = A[p+p*N];
	pivot = p;
      }
    }

    // Check exit conditions
    if (dmax < tol || dmax < 0.0)
      break;

    // Swap pivot indices
    int tempint = piv[i];
    piv[i] = pivot;
    piv[pivot] = tempint;
    
    // Swap pivot column
    memcpy(temparr, &A[i*N], N*sizeof(double));
    memcpy(&A[i*N], &A[pivot*N], N*sizeof(double));
    memcpy(&A[pivot*N], temparr, N*sizeof(double));
    
    // Swap pivot row
    for (int j = 0; j < N; j++) {
      double tempdouble = A[i+j*N];
      A[i+j*N] = A[pivot+j*N];
      A[pivot+j*N] = tempdouble;
    }
    
    // Scale by diagonal pivot
    double gii = sqrt(dmax);
    for (int r = i; r < N; r++) {
      A[r+i*N] /= gii;
    }

    // Update trailing submatrix
    if (i < N-1) {
      for (int c = i+1; c < N; c++) {
	for (int r = i+1; r < N; r++) {
	  A[r+c*N] -= A[r+i*N]*A[c+i*N];
	}
      }
    }
    i++;
  }
  *rank = i--;

  // Zero upper triangle
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      A[i+j*N] = 0.0;
    }
  }

  // Zero truncated portion of the matrix
  for (int i = 0; i < N; i++)
    for (int j = *rank; j < N; j++)
      A[i+j*N] = 0;
}

// Pivoted left-looking lazy Cholesky
void pll_cholesky(int N, double *A, double* G, int* piv, double tol, int* rank)
{
  // Matrix diagonal
  double* diag = (double*) malloc(N*sizeof(double));

  // Initialize pivot and set diag
  for (int i = 0; i < N; i++) {
    piv[i] = i;
    diag[i] = A[i+i*N];
  }

  // Begin main iteration
  int i = 0;
  while (i < N) {
    // Determine next pivot index
    int pivot = i;
    double dmax = diag[piv[i]];
    for (int p = i+1; p < N; p++) {
      if (dmax < diag[piv[p]]) {
	dmax = diag[piv[p]];
	pivot = p;
      }
    }

    // Check exit conditions
    if (dmax < tol || dmax < 0.0)
      break;
    
    // Swap pivot indices
    int tempint = piv[i];
    piv[i] = piv[pivot];
    piv[pivot] = tempint;
    
    // Compute diagonal of entry of Cholesky factor
    int gii = sqrt(dmax);
    G[piv[i]+i*N] = gii;

    // Compute column of original matrix
    for (int r = i+1; r < N; r++) {
      G[piv[r]+i*N] = A[piv[r]+piv[i]*N];
    }

    // Compute column of Cholesky factor
    for (int r = i+1; r < N; r++) {
      for (int c = 0; c < i; c++) {
	G[piv[r]+i*N] -= G[piv[i]+c*N]*G[piv[r]+c*N];
      }

      // Scale by diagonal pivot
      G[piv[r]+i*N] /= gii;

      // Update Schur complement diagonal
      diag[piv[r]] -= G[piv[r]+i*N]*G[piv[r]+i*N];
    }
    i++;
  }
  *rank = i--;
}

// Computes the trace norm error of the Cholesky factorization
void cholError(int N, double* A, double* G, double* error)
{
  for (int k = 0; k < N; k++) {
    double dotprod = 0;
    int imax = min(k + 1, N);
    for (int i = 0; i < imax; i++) {
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
  for (int k = 0; k < N; k++) {
    double dotprod = 0;
    for (int i = 0; i < N; i++) {
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
  for (int k = 0; k < N; k++) {
    double dotprod = 0;
    int imax = min(k + 1, N);
    for (int i = 0; i < imax; i++) {
      double prod = G[k + i * N] * G[k + i * N];
      dotprod += prod;
    }
    double Akk = A[piv[k] + piv[k] * N];
    (*error) += abs(Akk - dotprod);
  }
}

// Prints the matrix
void printMatrix(int N, double* A)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%2.1f, ", A[i+j*N]);
    }
    printf("\n");
  }
}

// Prints the matrix
void printDiag(int N, double* A)
{
  for (int i = 0; i < N; i++) {
    printf("%f\n", A[i+i*N]);
  }
}

// Evaluates the matrix from the Cholesky factor
void cholmatmul(int N, double* G, double* A, int rank)
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < rank; k++)
	A[i+j*N] += G[i+k*N]*G[j+k*N];
}
