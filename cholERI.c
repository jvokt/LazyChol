#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include "cholERI.h"
#include "cholesky.h"

// Computes the lazy pivoted Cholesky factorization of the ERI
void cholERI(BasisSet_t basis, ERD_t erd, double** G, double tol, int max_rank, int *rank)
{
  int n = CInt_getNumFuncs(basis);
  int N = n*n;
  max_rank = fmin(max_rank, N);

  // Preallocate space for Cholesky factor
  *G = (double*) calloc(N*max_rank,sizeof(double));

  // Preallocate matrix diagonal
  double* diag = (double*) malloc(N*sizeof(double));

  // Initialize pivot
  int* piv = (int*) malloc(N*sizeof(int));
  for (int i = 0; i < N; i++)
    piv[i] = i;
  
  // Compute diagonal
  computeDiag(basis, erd, diag); 

  // Begin main iteration
  int i = 0;
  while (i < max_rank) {
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
    
    // Compute column of original matrix
    double * column = *G + i*N;
    computeColumn(basis, erd, piv[i], column);
    
    // Zero unneeded column entries
    for (int k = 0; k < i; k++)
      (*G)[piv[k]+i*N] = 0.0;
    
    // Compute diagonal of entry of Cholesky factor
    double gii = sqrt(dmax);
    (*G)[piv[i]+i*N] = gii;
    
    // Compute column of Cholesky factor
    for (int r = i+1; r < N; r++) {
      for (int c = 0; c < i; c++) {
	(*G)[piv[r]+i*N] -= (*G)[piv[i]+c*N]*(*G)[piv[r]+c*N];
      }

      // Scale by diagonal pivot
      (*G)[piv[r]+i*N] /= gii;

      // Update Schur complement diagonal
      diag[piv[r]] -= (*G)[piv[r]+i*N] * (*G)[piv[r]+i*N];
    }    
    i++;
  }
  *rank = i--;
  free(diag);
  free(piv);
}

// Precomputes the diagonal for pivoting
void computeDiag(BasisSet_t basis, ERD_t erd, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);

  // For all shells M and N
  for (int M = 0; M < shellCount; M++) {
    for (int N = 0; N < shellCount; N++) {
      double *integrals;
      int nints;

      // Compute Shell Quartet (M N | M N)
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, M, N, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);

      // For all computed integrals, transfer to target buffer
      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int ij = (om + mstart) + (on + nstart) * n;
	  int idx = om + nM * (on + nN * (om + nM * on));
	  target[ij] = integrals[idx];
	}
      }
    }
  }
}

int func_to_shell(BasisSet_t basis, int funcid)
{
  int shellCount = CInt_getNumShells(basis);
  int shellDim = 0;
  int shellStart = 0;
  for (int M = 0; M < shellCount; M++) {
    shellDim = CInt_getShellDim(basis, M);
    shellStart = CInt_getFuncStartInd(basis, M);
    if (funcid >= shellStart && funcid < shellStart + shellDim)
      return M;
  }
  return -1;
}

// Computes a specified column
void computeColumn(BasisSet_t basis, ERD_t erd, int column, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);
  
  // Determine r and s from specified column
  int r = column / n;
  int s = column % n;
  int R = func_to_shell(basis,r);
  int S = func_to_shell(basis,s);
  
  int nR = CInt_getShellDim (basis, R);
  int nS = CInt_getShellDim (basis, S);
  int rstart = CInt_getFuncStartInd (basis, R);
  int sstart = CInt_getFuncStartInd (basis, S);
  
  int or = r - rstart;
  int os = s - sstart;
  
  // For all shells M and N
  for (int M = 0; M < shellCount; M++) {
    for (int N = 0; N < shellCount; N++) {
      double *integrals;
      int nints;

      // Compute Shell Quartet (M N | R S)
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, R, S, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);
      
      // For all computed integrals, transfer to target buffer
      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int ij = (om + mstart) + (on + nstart) * n;
	  int idx = om + nM * (on + nN * (or + nR * os));
	  target[ij] = integrals[idx];
	}
      }
    }
  }  
}

// Computes a shell quartet from the Cholesky factor
CIntStatus_t cholComputeShellQuartet(BasisSet_t basis,
				     double* G, int r,
				     int M,
				     int N,
				     int P,
				     int Q,
				     double **cholintegrals,
				     int *cholnints )
{
  int n = CInt_getNumFuncs(basis);
  int n2 = n*n;

  // Number of basis functions in a shell
  int dimM = CInt_getShellDim (basis, M);
  int dimN = CInt_getShellDim (basis, N);
  int dimP = CInt_getShellDim (basis, P);
  int dimQ = CInt_getShellDim (basis, Q);

  // First basis function index in a shell
  int startM = CInt_getFuncStartInd (basis, M);
  int startN = CInt_getFuncStartInd (basis, N);
  int startP = CInt_getFuncStartInd (basis, P);
  int startQ = CInt_getFuncStartInd (basis, Q);
  double* G1 = (double*) malloc(sizeof(double)*dimM*dimN*r);
  double* G2 = (double*) malloc(sizeof(double)*dimP*dimQ*r);
  *cholintegrals = (double*) calloc(dimM*dimN*dimP*dimQ,sizeof(double));
  
  // Initialize temporary matrix G1 with rows of shells M and N
  for (int ir = 0; ir < r; ir++) {
    for (int iM = 0; iM < dimM; iM++) {
      for (int iN = 0; iN < dimN; iN++) {
	int from = startM + iM + (startN + iN)*n + ir*n2;
	int to = iM + dimM * (iN + dimN * ir);
	G1[to] = G[from];
      }
    }
  }

  // Initialize temporary matrix G2 with rows of shells P and Q
  for (int ir = 0; ir < r; ir++) {
    for (int iP = 0; iP < dimP; iP++) {
      for (int iQ = 0; iQ < dimQ; iQ++) {
	int from = startP + iP + (startQ + iQ) * n + ir*n2;
	int to = iP + dimP * (iQ + dimQ * ir);
	G2[to] = G[from];
      }
    }
  }
  
  // Compute each entry of the shell quartet by multiplying G1 and G2
  for (int iM = 0; iM < dimM; iM++) {
    for (int iN = 0; iN < dimN; iN++) {
      for (int iP = 0; iP < dimP; iP++) {
	for (int iQ = 0; iQ < dimQ; iQ++) {
	  for (int ir = 0; ir < r; ir++) {
	    int from1 = iM + dimM * (iN + dimN * ir);
	    int from2 = iP + dimP * (iQ + dimQ * ir);
	    int to = iM + dimM * (iN + dimN * (iP + dimP * iQ));
	    (*cholintegrals)[to] += G1[from1]*G2[from2];
	  }
	}
      }
    }
  }
  *cholnints = dimM*dimN*dimP*dimQ;
  free(G1);
  free(G2);
  return CINT_STATUS_SUCCESS;
}

// Computes a shell quartet from the Cholesky factor
CIntStatus_t cholComputeShellQuartet2(BasisSet_t basis,
				     double* G, int r,
				     int M,
				     int N,
				     int P,
				     int Q,
				     double **cholintegrals,
				     int *cholnints )
{
  int n = CInt_getNumFuncs(basis);
  int n2 = n*n;

  // Number of basis functions in a shell
  int dimM = CInt_getShellDim (basis, M);
  int dimN = CInt_getShellDim (basis, N);
  int dimP = CInt_getShellDim (basis, P);
  int dimQ = CInt_getShellDim (basis, Q);

  // First basis function index in a shell
  int startM = CInt_getFuncStartInd (basis, M);
  int startN = CInt_getFuncStartInd (basis, N);
  int startP = CInt_getFuncStartInd (basis, P);
  int startQ = CInt_getFuncStartInd (basis, Q);
  *cholintegrals = (double*) calloc(dimM*dimN*dimP*dimQ,sizeof(double));
  
  // cholintegrals[iM, iN, iP, iQ] = G[iM, iN, :] * G[iP, iQ, :]'
  for (int iM = 0; iM < dimM; iM++) {
    for (int iN = 0; iN < dimN; iN++) {
      for (int iP = 0; iP < dimP; iP++) {
	for (int iQ = 0; iQ < dimQ; iQ++) {
	  for (int ir = 0; ir < r; ir++) {
	    int from1 = startM + iM + (startN + iN) * n + ir*n2;
	    int from2 = startP + iP + (startQ + iQ) * n + ir*n2;
	    int to = iM + dimM * (iN + dimN * (iP + dimP * iQ));
	    (*cholintegrals)[to] += G[from1] * G[from2];
	  }
	}
      }
    }
  }
  *cholnints = dimM*dimN*dimP*dimQ;
  return CINT_STATUS_SUCCESS;
}

// Computes the lazy pivoted Cholesky factorization of the ERI
void structcholERI(BasisSet_t basis, ERD_t erd, double** G, double tol, int max_rank, int *rank)
{
  int n = CInt_getNumFuncs(basis);
  int N = (n*(n+1))/2;
  max_rank = fmin(max_rank, N);

  // Preallocate space for Cholesky factor
  *G = (double*) calloc(N*max_rank,sizeof(double));

  // Preallocate matrix diagonal
  double* diag = (double*) malloc(N*sizeof(double));

  // Initialize pivot
  int* piv = (int*) malloc(N*sizeof(int));
  for (int i = 0; i < N; i++)
    piv[i] = i;
  
  // Compute diagonal
  computeStructDiag(basis, erd, diag);

  // Begin main iteration
  int i = 0;
  while (i < max_rank) {
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

    // Compute column of original matrix
    double * column = *G + i*N;
    computeStructColumn(basis, erd, piv[i], column);
    
    // Zero unneeded column entries
    for (int k = 0; k < i; k++)
      (*G)[piv[k]+i*N] = 0.0;
    
    // Compute diagonal of entry of Cholesky factor
    double gii = sqrt(dmax);
    (*G)[piv[i]+i*N] = gii;

    // Compute column of Cholesky factor
    for (int r = i+1; r < N; r++) {
      for (int c = 0; c < i; c++) {
	(*G)[piv[r]+i*N] -= (*G)[piv[i]+c*N]*(*G)[piv[r]+c*N];
      }
      
      // Scale by diagonal pivot
      (*G)[piv[r]+i*N] /= gii;
      
      // Update Schur complement diagonal
      diag[piv[r]] -= (*G)[piv[r]+i*N]* (*G)[piv[r]+i*N];
    }
    i++;
  }
  *rank = i--;
  free(diag);
  free(piv);
}

// Precomputes the diagonal for pivoting (structured)
void computeStructDiag(BasisSet_t basis, ERD_t erd, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);

  // For all shells M < N
  for (int M = 0; M < shellCount; M++) {
    for (int N = M+1; N < shellCount; N++) {
      double *integrals;
      int nints;

      // Compute Shell Quartet (M N | M N)
      CInt_computeShellQuartet(basis, erd, 0,
			       M, N, M, N, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);

      // For all computed integrals, transfer to target buffer
      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int i = on + nstart;
	  int j = om + mstart;

	  // Used packed index to exploit symmetry
	  int ij = i - (j * (j + 1)) / 2 + j * n;
	  int idx = om + nM * (on + nN * (om + nM * on));
	  target[ij] = integrals[idx];
	}
      }
    }
  }  

  // For all shells M
  for (int M = 0; M < shellCount; M++) {
    double *integrals;
    int nints;
    
    // Compute Shell Quartet (M M | M M)
    CInt_computeShellQuartet(basis, erd, 0, 
			     M, M, M, M, &integrals, &nints);
    int nM = CInt_getShellDim (basis, M);
    int mstart = CInt_getFuncStartInd (basis, M);
    
    // Transfer integrals on lower triangle to target buffer
    for (int om = 0; om < nM; om++) {
      for (int on = om; on < nM; on++) {
	int i = on + mstart;
	int j = om + mstart;
	
	// Use packed index to exploit symmetry
	int ij = i - (j * (j + 1)) / 2 + j * n;
	int idx = om + nM * (on + nM * (om + nM * on));
	target[ij] = integrals[idx];
      }
    }
  }
}

/*    Returns the indices (i,j) corresponding to packed index c
 *    Assume that c = i-(1/2)(j-1)j+(j-1)n is between 1 and n(n+1)/2
 *    where j is between 1 and n, and i is between j and n
 *    Indexing from zero: c0 = i0-(1/2)j0(j0+1)+j0n, i0 >= j0
 *    If i0 == j0, then c0 is quadratic in one variable k0 == i0 == j0
 *    If i0 > j0 and we can find j0, all we need is the offset i0-j0 to find i0
 *    Note: there is an interval of c0's with contant j0 (called interval-j0)
 *    and i0 is determined by the offset within interval-j0
 *    c0 = c-1
 *    Solve the quadratic equation k0^2-(2n+1)k0+2c0 = 0 for k0
 */
void unpack_index(int c0, int n, int *i0, int *j0) {
  double p = pow(2 * n + 1, 2);
  double s = sqrt(p - 8 * c0);
  double k0 = (2 * n + 1 - s) / 2;
  //    Interval-j0 is the integer component of k0
  *j0 = floor(k0);
  //    Let the first value in interval-j0 be called d0
  int d0 = *j0 - (*j0 * (*j0 + 1)) / 2 + *j0 * n;
  //    The offset i0-j0 is simply c0-d0
  *i0 = *j0 + (c0 - d0);
}

// Computes a specified column (structured)
void computeStructColumn(BasisSet_t basis, ERD_t erd, int column, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);

  // Determine r and s from specified column
  int r;
  int s;
  unpack_index(column, n, &s, &r);
  int R = func_to_shell(basis,r);
  int S = func_to_shell(basis,s);
  
  int nR = CInt_getShellDim (basis, R);
  int nS = CInt_getShellDim (basis, S);
  int rstart = CInt_getFuncStartInd (basis, R);
  int sstart = CInt_getFuncStartInd (basis, S);
  
  int or = r - rstart;
  int os = s - sstart;

  // For all shells M < N
  for (int M = 0; M < shellCount; M++) {
    for (int N = M+1; N < shellCount; N++) {
      double *integrals;
      int nints;

      // Compute Shell Quartet (M N | R S)
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, R, S, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);
      
      // For all computed integrals, transfer to target buffer
      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int i = on + nstart;
	  int j = om + mstart;

	  // Use packed index to exploit symmetry
	  int ij = i - (j * (j + 1)) / 2 + j * n;
	  int idx = om + nM * (on + nN * (or + nR * os));
	  target[ij] = integrals[idx];
	}
      }
    }
  }  

  // For all shells M
  for (int M = 0; M < shellCount; M++) {
    double *integrals;
    int nints;

    // Compute Shell Quartet (M M | R S)
    CInt_computeShellQuartet(basis, erd, 0, 
			     M, M, R, S, &integrals, &nints);
    int nM = CInt_getShellDim (basis, M);
    int mstart = CInt_getFuncStartInd (basis, M);
    
    // Transfer integrals on lower triangle to target buffer
    for (int om = 0; om < nM; om++) {
      for (int on = om; on < nM; on++) {
	int i = on + mstart;
	int j = om + mstart;

	// Use packed index to exploit symmetry
	int ij = i - (j * (j + 1)) / 2 + j * n;
	int idx = om + nM * (on + nM * (or + nR * os));
	target[ij] = integrals[idx];
      }
    }
  }
}

// Computes a shell quartet from the structured Cholesky factor
CIntStatus_t structcholComputeShellQuartet(BasisSet_t basis,
				     double* G, int r,
				     int M,
				     int N,
				     int P,
				     int Q,
				     double **cholintegrals,
				     int *cholnints )
{
  int n = CInt_getNumFuncs(basis);
  int Ndim = (n*(n+1))/2;

  // Number of basis functions in a shell
  int dimM = CInt_getShellDim (basis, M);
  int dimN = CInt_getShellDim (basis, N);
  int dimP = CInt_getShellDim (basis, P);
  int dimQ = CInt_getShellDim (basis, Q);

  // First basis function index in a shell
  int startM = CInt_getFuncStartInd (basis, M);
  int startN = CInt_getFuncStartInd (basis, N);
  int startP = CInt_getFuncStartInd (basis, P);
  int startQ = CInt_getFuncStartInd (basis, Q);
  double* G1 = (double*) malloc(sizeof(double)*dimM*dimN*r);
  double* G2 = (double*) malloc(sizeof(double)*dimP*dimQ*r);
  *cholintegrals = (double*) calloc(dimM*dimN*dimP*dimQ,sizeof(double));
  
  // Initialize temporary matrix G1 with rows of shells M and N
  for (int ir = 0; ir < r; ir++) {
    for (int iM = 0; iM < dimM; iM++) {
      for (int iN = 0; iN < dimN; iN++) {
	int i = fmax(startM + iM, startN + iN);
	int j = fmin(startM + iM, startN + iN);
	int from = i - (j * (j + 1)) / 2 + j * n + ir*Ndim;
	int to = iM + dimM * (iN + dimN * ir);
	G1[to] = G[from];
      }
    }
  }

  // Initialize temporary matrix G2 with rows of shells P and Q
  for (int ir = 0; ir < r; ir++) {
    for (int iP = 0; iP < dimP; iP++) {
      for (int iQ = 0; iQ < dimQ; iQ++) {
	int i = fmax(startP + iP, startQ + iQ);
	int j = fmin(startP + iP, startQ + iQ);
	int from = i - (j * (j + 1)) / 2 + j * n + ir*Ndim;
	int to = iP + dimP * (iQ + dimQ * ir);
	G2[to] = G[from];
      }
    }
  }
  
  // Compute each entry of the shell quartet by multiplying G1 and G2
  for (int iM = 0; iM < dimM; iM++) {
    for (int iN = 0; iN < dimN; iN++) {
      for (int iP = 0; iP < dimP; iP++) {
	for (int iQ = 0; iQ < dimQ; iQ++) {
	  for (int ir = 0; ir < r; ir++) {
	    int from1 = iM + dimM * (iN + dimN * ir);
	    int from2 = iP + dimP * (iQ + dimQ * ir);
	    int to = iM + dimM * (iN + dimN * (iP + dimP * iQ));
	    (*cholintegrals)[to] += G1[from1]*G2[from2];
	  }
	}
      }
    }
  }
  *cholnints = dimM*dimN*dimP*dimQ;
  free(G1);
  free(G2);
  return CINT_STATUS_SUCCESS;
}

// Computes a shell quartet from the structured Cholesky factor
CIntStatus_t structcholComputeShellQuartet2(BasisSet_t basis,
				     double* G, int r,
				     int M,
				     int N,
				     int P,
				     int Q,
				     double **cholintegrals,
				     int *cholnints )
{
  int n = CInt_getNumFuncs(basis);
  int n2 = n*n;

  // Number of basis functions in a shell
  int dimM = CInt_getShellDim (basis, M);
  int dimN = CInt_getShellDim (basis, N);
  int dimP = CInt_getShellDim (basis, P);
  int dimQ = CInt_getShellDim (basis, Q);

  // First basis function index in a shell
  int startM = CInt_getFuncStartInd (basis, M);
  int startN = CInt_getFuncStartInd (basis, N);
  int startP = CInt_getFuncStartInd (basis, P);
  int startQ = CInt_getFuncStartInd (basis, Q);
  *cholintegrals = (double*) calloc(dimM*dimN*dimP*dimQ,sizeof(double));
  
  // cholintegrals[iM, iN, iP, iQ] = G[iM, iN, :] * G[iP, iQ, :]'
  for (int iM = 0; iM < dimM; iM++) {
    for (int iN = 0; iN < dimN; iN++) {
      for (int iP = 0; iP < dimP; iP++) {
	for (int iQ = 0; iQ < dimQ; iQ++) {
	  for (int ir = 0; ir < r; ir++) {
	    int i1 = fmax(startP + iP, startQ + iQ);
	    int j1 = fmin(startP + iP, startQ + iQ);
	    int from1 = i1 - (j1*(j1 + 1)) / 2 + j1 * n + ir*n2;
	    int i2 = fmax(startP + iP, startQ + iQ);
	    int j2 = fmin(startP + iP, startQ + iQ);
	    int from2 = i2 - (j2*(j2 + 1)) / 2 + j2 * n + ir*n2;
	    int to = iM + dimM * (iN + dimN * (iP + dimP * iQ));
	    (*cholintegrals)[to] += G[from1] * G[from2];
	  }
	}
      }
    }
  }
  *cholnints = dimM*dimN*dimP*dimQ;
  return CINT_STATUS_SUCCESS;
}
