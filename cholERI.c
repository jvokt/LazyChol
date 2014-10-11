#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include "cholERI.h"
#include "cholesky.h"

// Computes the lazy pivoted Cholesky factorization of the ERI
void cholERI(BasisSet_t basis, ERD_t erd, double** G, double tol, int *rank)
{
  int n = CInt_getNumFuncs(basis);
  int n2 = n*n;
  int N = n2;
  int max_rank = n2;//(1-floor(log10(tol)))*n;
  *G = (double*) calloc(N*max_rank,sizeof(double));
  double* diag = (double*) malloc(N*sizeof(double));
  int* piv = (int*) malloc(N*sizeof(int));
  for (int i = 0; i < N; i++)
    piv[i] = i;
  
  // compute diagonal
  printf("Computed diagonal\n");
  computeDiag(basis, erd, diag);
  for (int i = 0; i < N; i++)
    printf("A[%d,%d]=%f\n",i,i,diag[i]);
  int i, p, j, c, r, pivot, tempint;
  double tempdouble, gii, dmax;
  //  printf("Begin while loop\n");
  i = 0;
  while (i < max_rank) {
    //    printf("i: %d\n",i);
    pivot = i;
    dmax = diag[piv[i]];
    for (p = i+1; p < N; p++) {
      if (dmax < diag[piv[p]]) {
	dmax = diag[piv[p]];
	pivot = p;
      }
    }
    //    printf("Max: %f\n",dmax);
    //    printf("Pivot: %d\n",pivot);
    // check exit conditions
    if (dmax < tol || dmax < 0.0)
      break;
    
    // swap pivot
    tempint = piv[i];
    piv[i] = piv[pivot];
    piv[pivot] = tempint;
    
    // scale by diagonal pivot
    gii = sqrt(dmax);
    printf("i: %d, piv[i]: %d, gii: %1.2e; ",i,piv[i],gii);
    //    printf("Compute matrix column\n");
    // compute column of original matrix
    double * column = *G + i*N;
    computeColumn(basis, erd, piv[i], column);
    // Zero the upper triangle
    for (int k = 0; k < i; k++)
      (*G)[piv[k]+i*N] = 0.0;
    // set the pivot
    (*G)[piv[i]+i*N] = gii;
    
    for (int c=0; c<N; c++)
      printf("G[%d,%d]=%1.2e\n", c,i,(*G)[c+i*N]);
    //    for (r = i+1; r < N; r++) {
    //      G[piv[r]+i*N] = A[piv[r]+piv[i]*N];
    //    }
    //    printf("Set diagonal pivot G[%d+%d*N]=%2.1f\n",piv[i],i,gii);
    //    printf("Compute Cholesky column\n");
    // compute column of cholesky factor
    for (r = i+1; r < N; r++) {
      for (c = 0; c < i; c++) {
	(*G)[piv[r]+i*N] -= (*G)[piv[i]+c*N]*(*G)[piv[r]+c*N];
      }
      printf("G[%d,%d]*G[%d,%d]: %1.2e; ",piv[i],i,piv[r],i,(*G)[piv[r]+i*N]);
      (*G)[piv[r]+i*N] /= gii;
      printf("G[%d,%d]: %1.2e; ",piv[r],i,(*G)[piv[r]+i*N]);
      // update Schur complement diagonal
      diag[piv[r]] -= (*G)[piv[r]+i*N] * (*G)[piv[r]+i*N];
    }    
    //    for (int c=0; c<N; c++)
    //      printf("%2.1f\n",(*G)[c+i*N]);
    
    i++;
    printf("\n");
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
  for (int M = 0; M < shellCount; M++) {
    for (int N = 0; N < shellCount; N++) {
      double *integrals;
      int nints;
      
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, M, N, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);

      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  double ivalue = integrals[om + nM * (on + nN * (om + nM * on))];	  
	  target[(om + mstart) + (on + nstart) * n] = ivalue;

	    //   integrals[om * nN * nM * nN + on * nM * nN + om * nN + on];
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
      {
	//	printf("shellStart: %d, funcid: %d, shellStart+shellDim: %d, nextShellStart: %d\n",shellStart, funcid, shellStart+shellDim, CInt_getFuncStartInd(basis, M+1));
      return M;
      }
  }
  return -1;
}

// Computes a specified column
void computeColumn(BasisSet_t basis, ERD_t erd, int column, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);
  
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
  
  for (int M = 0; M < shellCount; M++) {
    for (int N = 0; N < shellCount; N++) {
      double *integrals;
      int nints;
      
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, R, S, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);
      
      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int idx = (om + mstart) + (on + nstart) * n;
	  //	  printf("column idx: %d\n",idx);
	  //	  double ivalue = integrals[om * nN * nR * nS + on * nR * nS + oR * nS + os];
	  double ivalue = integrals[om + nM * (on + nN * (or + nR * os))];

	  //	  printf("ivalue: %2.1f\n",ivalue);
	  target[idx] = ivalue;
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
  // find rows of the Cholesky factor which correspond to shells M, N, P and Q
  int n = CInt_getNumFuncs(basis);
  int n2 = n*n;
  int dimM = CInt_getShellDim (basis, M);
  int dimN = CInt_getShellDim (basis, N);
  int dimP = CInt_getShellDim (basis, P);
  int dimQ = CInt_getShellDim (basis, Q);
  int startM = CInt_getFuncStartInd (basis, M);
  int startN = CInt_getFuncStartInd (basis, N);
  int startP = CInt_getFuncStartInd (basis, P);
  int startQ = CInt_getFuncStartInd (basis, Q);
  // integrals (ij|kl) are such that:
  // i = startM : startM+dimM-1, 
  // j = startN : startN+dimN-1, 
  // k = startP : startP+dimP-1, 
  // l = startQ : startQ+dimQ-1
  // r = i + j*n, c = k + l*n
  //  printf("Allocate G1[%d]\n",dimM*dimN*r);
  //  double* G1 = (double*) malloc(sizeof(double)*dimM*dimN*r);
  //  printf("Allocate G2[%d]\n",dimP*dimQ*r);
  //  double* G2 = (double*) malloc(sizeof(double)*dimP*dimQ*r);
  //  printf("Generate G1\n");
  /*
  for (int ir = 0; ir < r; ir++) {
    for (int iM = 0; iM < dimM; iM++) {
      for (int iN = 0; iN < dimN; iN++) {
	//	int idx = startM + iM + dimM * (startN + iN + dimN * ir);
	int idx = startM + iM + (startN + iN)*n + ir*n2;
	//	printf("idx: %d\n",idx);
	G1[iM + dimM * (iN + dimN * ir)] = G[idx];
      }
    }
  }
  */
  /*
  for (int i = 0; i < dimM*dimN; i++) {
    for (int j = 0; j < r; j++)
      printf("%2.1f, ",G1[i,j]);
    printf("\n");
  }
  */
  /*
  //  printf("Generate G2\n");
  for (int ir = 0; ir < r; ir++) {
    for (int iP = 0; iP < dimP; iP++) {
      for (int iQ = 0; iQ < dimQ; iQ++) {
	int idx = startP + iP + (startQ + iQ) * n + ir*n2;
	//	printf("get val idx: %d\n",startP + iP + (startQ + iQ) * n + ir*n2);
	double val = G[idx];
	int target = iP + dimP * (iQ + dimQ * ir);
	//	printf("set val target: %d\n",target);
	G2[target] = val;
      }
    }
  }
  */
  /*
  for (int i = 0; i < dimP*dimQ; i++) {
    for (int j = 0; j < r; j++)
      printf("%2.1f, ",G2[i,j]);
    printf("\n");
  }
  */
  //  printf("Allocate cholintegrals[%d]\n",dimM*dimN*dimP*dimQ);
  *cholintegrals = (double*) calloc(dimM*dimN*dimP*dimQ,sizeof(double));
  // return the integrals in cholintegrals
  //  dgemm('N', 'T', dimM*dimN, dimP*dimQ, r, 1, G1, dimM*dimN,
  //	G2, dimM*dimN, 0, *cholintegrals, dimM*dimN);
  
  /*
  for (int i = 0; i < dimM*dimN; i++)
    for (int j = 0; j < dimP*dimQ; j++)
      for (int k = 0; k < r; k++) {
	int G1idx = i+dimM*dimN*k;
	int G2idx = j+dimP*dimQ*k;
	int cholintidx = i+dimM*dimN*j;
	//	printf("G1idx: %d\n",i+dimM*dimN*k);
	//	printf("G2idx: %d\n", j+dimP*dimQ*k);
	//	printf("cholintidx: %d\n", i+dimM*dimN*j);
	(*cholintegrals)[cholintidx] += G1[G2idx]*G2[G1idx];
      }
  */
  
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
  
  //  printf("Finished");
  *cholnints = dimM*dimN*dimP*dimQ;
  //  free(G1);
  //  free(G2);
  return CINT_STATUS_SUCCESS;
}

// Computes the lazy pivoted Cholesky factorization of the ERI
void structcholERI(BasisSet_t basis, ERD_t erd, double** G, double tol, int *rank)
{
  int n = CInt_getNumFuncs(basis);
  int n2 = n*n;
  int N = (n*(n+1))/2;
  int max_rank = N;//(1-floor(log10(tol)))*n;
  *G = (double*) calloc(N*max_rank,sizeof(double));
  double* diag = (double*) malloc(N*sizeof(double));
  int* piv = (int*) malloc(N*sizeof(int));
  for (int i = 0; i < N; i++)
    piv[i] = i;
  
  // compute diagonal
  //  printf("Compute diagonal\n");
  computeStructDiag(basis, erd, diag);
  for (int i = 0; i < N; i++)
    printf("%2.1f\n",diag[i]);
  int i, p, j, c, r, pivot, tempint;
  double tempdouble, gii, dmax;
  //  printf("Begin while loop\n");
  i = 0;
  while (i < max_rank) {
    //    printf("i: %d\n",i);
    pivot = i;
    dmax = diag[piv[i]];
    for (p = i+1; p < N; p++) {
      if (dmax < diag[piv[p]]) {
	dmax = diag[piv[p]];
	pivot = p;
      }
    }
    //    printf("Max: %f\n",dmax);
    //    printf("Pivot: %d\n",pivot);
    // check exit conditions
    if (dmax < tol || dmax < 0.0)
      break;
    
    // swap pivot
    tempint = piv[i];
    piv[i] = piv[pivot];
    piv[pivot] = tempint;
    
    // scale by diagonal pivot
    gii = sqrt(dmax);
    //    printf("Compute matrix column\n");
    // compute column of original matrix
    double * column = *G + i*N;
    computeStructColumn(basis, erd, pivot, column);
    
    //    for (int c=0; c<N; c++)
    //      printf("%2.1f\n", (*G)[c+i*N]);
    //    for (r = i+1; r < N; r++) {
    //      G[piv[r]+i*N] = A[piv[r]+piv[i]*N];
    //    }
    //    printf("Set diagonal pivot G[%d+%d*N]=%2.1f\n",piv[i],i,gii);
    //    printf("Compute Cholesky column\n");
    // compute column of cholesky factor
    for (r = i+1; r < N; r++) {
      for (c = 0; c < i; c++) {
	(*G)[piv[r]+i*N] -= (*G)[piv[i]+c*N]*(*G)[piv[r]+c*N];
      }
      (*G)[piv[r]+i*N] /= gii;
      // update Schur complement diagonal
      diag[piv[r]] -= (*G)[piv[r]+i*N]* (*G)[piv[r]+i*N];
    }
    
    // Zero the upper triangle
    for (int k = 0; k < i; k++)
      (*G)[piv[k]+i*N] = 0.0;
    
    (*G)[piv[i]+i*N] = gii;
    
    //    for (int c=0; c<N; c++)
    //      printf("%2.1f\n",(*G)[c+i*N]);
    
    i++;
  }
  *rank = i--;
  free(diag);
  free(piv);
  /*
  int nbf = CInt_getNumFuncs(basis);
  int N = (nbf*(nbf+1))/2;
  int max_num_cols = 7*nbf;
  double* diag = (double*) malloc(sizeof(double)*N);
  int* pivot = (double*) malloc(sizeof(int)*N);
  double* G = (double*) malloc(sizeof(double)*N*max_num_cols);
  // compute diagonal
  computeDiag(basis, erd, diag);
  // while true
  for (int i = 0; i < max_num_cols; i++) {
    // check exit conditions
    // swap pivot
    // set the diagonal of the Cholesky factor
    // compute column
    // gaxpy update column
    // divide by diagonal of the Cholesky factor
    // update the diagonal
    // end while
  }
  */
}

// Precomputes the diagonal for pivoting (structured)
void computeStructDiag(BasisSet_t basis, ERD_t erd, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);
  for (int M = 0; M < shellCount; M++) {
    for (int N = M+1; N < shellCount; N++) {
      double *integrals;
      int nints;
      
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, M, N, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);

      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int i = on + nstart;
	  int j = om + mstart;
	  int ij = i - (j * (j + 1)) / 2 + j * n;
	  int bid = om * nN * nM * nN + on * nM * nN + om * nN + on;
	  target[ij] = integrals[bid];
	}
      }
    }
  }  
  for (int M = 0; M < shellCount; M++) {
    double *integrals;
    int nints;
    
    CInt_computeShellQuartet(basis, erd, 0, 
			     M, M, M, M, &integrals, &nints);
    int nM = CInt_getShellDim (basis, M);
    int nM2 = nM * nM;
    int nM3 = nM * nM2;
    int mstart = CInt_getFuncStartInd (basis, M);
    
    for (int om = 0; om < nM; om++) {
      for (int on = om; on < nM; on++) {
	int i = on + mstart;
	int j = om + mstart;
	int ij = i - (j * (j + 1)) / 2 + j * n;
	int bid = om * nM3 + on * nM2 + om * nM + on;
	target[ij] = integrals[bid];
      }
    }
  }
  /*
  const double* buffer = integral_->buffer();
  int n = basisset_->nbf();
  int i;
  int j;
  int ij;
  int bid;
  for (int M = 0; M < basisset_->nshell(); M++) {
    for (int N = M + 1; N < basisset_->nshell(); N++) {

      integral_->compute_shell(M, N, M, N);

      int nM = basisset_->shell(M).nfunction();
      int nN = basisset_->shell(N).nfunction();
      int mstart = basisset_->shell(M).function_index();
      int nstart = basisset_->shell(N).function_index();

      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  i = on + nstart;
	  j = om + mstart;
	  ij = i - (j * (j + 1)) / 2 + j * n;
	  bid = om * nN * nM * nN + on * nM * nN + om * nN + on;
	  target[ij] = buffer[bid];
	}
      }
    }
  }
  for (int M = 0; M < basisset_->nshell(); M++) {

    integral_->compute_shell(M, M, M, M);

    int nM = basisset_->shell(M).nfunction();
    int nM2 = nM * nM;
    int nM3 = nM * nM2;
    int mstart = basisset_->shell(M).function_index();

    for (int om = 0; om < nM; om++) {
      for (int on = om; on < nM; on++) {
	i = on + mstart;
	j = om + mstart;
	ij = i - (j * (j + 1)) / 2 + j * basisset_->nbf();
	bid = om * nM3 + on * nM2 + om * nM + on;
	target[ij] = buffer[bid];
      }
    }
  }  
  */
}
void unpack_index(int c0, int n, int *i0, int *j0) {
  //    Returns the indices (i,j) corresponding to packed index c
  //    Assume that c = i-(1/2)(j-1)j+(j-1)n is between 1 and n(n+1)/2
  //    where j is between 1 and n, and i is between j and n
  //    Indexing from zero: c0 = i0-(1/2)j0(j0+1)+j0n, i0 >= j0
  //    If i0 == j0, then c0 is quadratic in one variable k0 == i0 == j0
  //    If i0 > j0 and we can find j0, all we need is the offset i0-j0 to find i0
  //    Note: there is an interval of c0's with contant j0 (called interval-j0)
  //    and i0 is determined by the offset within interval-j0
  //    c0 = c-1
  //    Solve the quadratic equation k0^2-(2n+1)k0+2c0 = 0 for k0
  double p = pow(2 * n + 1, 2);
  double s = sqrt(p - 8 * c0);
  double k0 = (2 * n + 1 - s) / 2;
  //    Interval-j0 is the integer component of k0
  *j0 = floor(k0);
  //    Let the first value in interval-j0 be called d0
  int d0 = *j0 - (*j0 * (*j0 + 1)) / 2 + *j0 * n;
  //    printf("d0 = %d\n",d0);
  //    The offset i0-j0 is simply c0-d0
  *i0 = *j0 + (c0 - d0);
  //    return (i0+1,j0+1)
}

// Computes a specified column (structured)
void computeStructColumn(BasisSet_t basis, ERD_t erd, int column, double* target)
{
  int n = CInt_getNumFuncs(basis);
  int shellCount = CInt_getNumShells(basis);

  int r;
  int s;
  unpack_index(column, n, &s, &r);
  int R = func_to_shell(basis,r);
  int S = func_to_shell(basis,s);
  
  int nR = CInt_getShellDim (basis, R);
  int nS = CInt_getShellDim (basis, S);
  int rstart = CInt_getFuncStartInd (basis, R);
  int sstart = CInt_getFuncStartInd (basis, S);
  
  int oR = r - rstart;
  int os = s - sstart;

  for (int M = 0; M < shellCount; M++) {
    for (int N = M+1; N < shellCount; N++) {
      double *integrals;
      int nints;
      
      CInt_computeShellQuartet(basis, erd, 0, 
			       M, N, R, S, &integrals, &nints);
      int nM = CInt_getShellDim (basis, M);
      int nN = CInt_getShellDim (basis, N);
      int mstart = CInt_getFuncStartInd (basis, M);
      int nstart = CInt_getFuncStartInd (basis, N);
      
      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  int i = on + nstart;
	  int j = om + mstart;
	  int ij = i - (j * (j + 1)) / 2 + j * n;
	  int bid = om * nN * nR * nS + on * nR * nS + oR * nS + os;
	  target[ij] = integrals[bid];
	}
      }
    }
  }  
  for (int M = 0; M < shellCount; M++) {
    double *integrals;
    int nints;
    
    CInt_computeShellQuartet(basis, erd, 0, 
			     M, M, R, S, &integrals, &nints);
    int nM = CInt_getShellDim (basis, M);
    int nM2 = nM * nM;
    int nM3 = nM * nM2;
    int mstart = CInt_getFuncStartInd (basis, M);
    
    for (int om = 0; om < nM; om++) {
      for (int on = om; on < nM; on++) {
	int i = on + mstart;
	int j = om + mstart;
	int ij = i - (j * (j + 1)) / 2 + j * n;
	int bid = om * nM * nR * nS + on * nR * nS + oR * nS + os;
	target[ij] = integrals[bid];
      }
    }
  }
  /*
  int n = basisset_->nbf();
  int s;
  int r;

  const double* buffer = integral_->buffer();

  unpack_index(row, n, s, r);
  int R = basisset_->function_to_shell(r);
  int S = basisset_->function_to_shell(s);

  int nR = basisset_->shell(R).nfunction();
  int nS = basisset_->shell(S).nfunction();
  int rstart = basisset_->shell(R).function_index();
  int sstart = basisset_->shell(S).function_index();

  int oR = r - rstart;
  int os = s - sstart;

  int i;
  int j;
  int ij;
  for (int M = 0; M < basisset_->nshell(); M++) {
    for (int N = M + 1; N < basisset_->nshell(); N++) {

      integral_->compute_shell(M, N, R, S);

      int nM = basisset_->shell(M).nfunction();
      int nN = basisset_->shell(N).nfunction();
      int mstart = basisset_->shell(M).function_index();
      int nstart = basisset_->shell(N).function_index();

      for (int om = 0; om < nM; om++) {
	for (int on = 0; on < nN; on++) {
	  i = on + nstart;
	  j = om + mstart;
	  ij = i - (j * (j + 1)) / 2 + j * basisset_->nbf();
	  target[ij] = buffer[om * nN * nR * nS + on * nR * nS
			      + oR * nS + os];
	}
      }
    }
  }

  for (int M = 0; M < basisset_->nshell(); M++) {

    integral_->compute_shell(M, M, R, S);

    int nM = basisset_->shell(M).nfunction();
    int mstart = basisset_->shell(M).function_index();

    for (int om = 0; om < nM; om++) {
      for (int on = om; on < nM; on++) {
	i = on + mstart;
	j = om + mstart;
	ij = i - (j * (j + 1)) / 2 + j * basisset_->nbf();
	target[ij] = buffer[om * nM * nR * nS + on * nR * nS + oR * nS
			    + os];
      }
    }
  }  
  */
}


// Computes a shell quartet from the Cholesky factor
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
  int n2 = n*n;
  int dimM = CInt_getShellDim (basis, M);
  int dimN = CInt_getShellDim (basis, N);
  int dimP = CInt_getShellDim (basis, P);
  int dimQ = CInt_getShellDim (basis, Q);
  int startM = CInt_getFuncStartInd (basis, M);
  int startN = CInt_getFuncStartInd (basis, N);
  int startP = CInt_getFuncStartInd (basis, P);
  int startQ = CInt_getFuncStartInd (basis, Q);
  // integrals (ij|kl) are such that:
  // i = startM : startM+dimM-1, 
  // j = startN : startN+dimN-1, 
  // k = startP : startP+dimP-1, 
  // l = startQ : startQ+dimQ-1
  // r = i + j*n, c = k + l*n
  //  printf("Allocate G1[%d]\n",dimM*dimN*r);
  double* G1 = (double*) malloc(sizeof(double)*dimM*dimN*r);
  //  printf("Allocate G2[%d]\n",dimP*dimQ*r);
  double* G2 = (double*) malloc(sizeof(double)*dimP*dimQ*r);
  //  printf("Generate G1\n");
  for (int ir = 0; ir < r; ir++) {
    for (int iM = 0; iM < dimM; iM++) {
      for (int iN = 0; iN < dimN; iN++) {
	//	int idx = startM + iM + dimM * (startN + iN + dimN * ir);
	int i = fmax(startM + iM, startN + iN);
	int j = fmin(startM + iM, startN + iN);
	int idx = i - (j * (j + 1)) / 2 + j * n + ir*n2;
	//	printf("idx: %d\n",idx);
	G1[iM + dimM * (iN + dimN * ir)] = G[idx];
      }
    }
  }
  /*
  for (int i = 0; i < dimM*dimN; i++) {
    for (int j = 0; j < r; j++)
      printf("%2.1f, ",G1[i,j]);
    printf("\n");
  }
  */
  //  printf("Generate G2\n");
  for (int ir = 0; ir < r; ir++) {
    for (int iP = 0; iP < dimP; iP++) {
      for (int iQ = 0; iQ < dimQ; iQ++) {
	//int idx = startP + iP + (startQ + iQ) * n + ir*n2;
	int i = fmax(startP + iP, startQ + iQ);
	int j = fmin(startP + iP, startQ + iQ);
	int idx = i - (j * (j + 1)) / 2 + j * n + ir*n2;	
	//	printf("get val idx: %d\n",startP + iP + (startQ + iQ) * n + ir*n2);
	double val = G[idx];
	int target = iP + dimP * (iQ + dimQ * ir);
	//	printf("set val target: %d\n",target);
	G2[target] = val;
      }
    }
  }
  /*
  for (int i = 0; i < dimP*dimQ; i++) {
    for (int j = 0; j < r; j++)
      printf("%2.1f, ",G2[i,j]);
    printf("\n");
  }
  */
  //  printf("Allocate cholintegrals[%d]\n",dimM*dimN*dimP*dimQ);
  *cholintegrals = (double*) calloc(dimM*dimN*dimP*dimQ,sizeof(double));
  // return the integrals in cholintegrals
  //  dgemm('N', 'T', dimM*dimN, dimP*dimQ, r, 1, G1, dimM*dimN,
  //	G2, dimM*dimN, 0, *cholintegrals, dimM*dimN);
  for (int i = 0; i < dimM*dimN; i++)
    for (int j = 0; j < dimP*dimQ; j++)
      for (int k = 0; k < r; k++) {
	int G1idx = i+dimM*dimN*k;
	int G2idx = j+dimP*dimQ*k;
	int cholintidx = i+dimM*dimN*j;
	//	printf("G1idx: %d\n",i+dimM*dimN*k);
	//	printf("G2idx: %d\n", j+dimP*dimQ*k);
	//	printf("cholintidx: %d\n", i+dimM*dimN*j);
	(*cholintegrals)[cholintidx] += G1[G2idx]*G2[G1idx];
      }
  //  printf("Finished");
  *cholnints = dimM*dimN*dimP*dimQ;
  free(G1);
  free(G2);
  return CINT_STATUS_SUCCESS;
}
