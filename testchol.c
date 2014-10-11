#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cholERI.h"
#include "CInt.h"

int main (int argc, char **argv)
{
  double tol = 1e-6;
  if (argc != 3) {
    printf ("Usage: %s <basisset> <xyz>\n", argv[0]);
    return -1;
  }
  
  // load basis set
  BasisSet_t basis;
  CInt_createBasisSet(&basis);
  CInt_loadBasisSet(basis, argv[1], argv[2]);
  
  printf("Molecule info:\n");
  printf("  #Atoms\t= %d\n", CInt_getNumAtoms(basis));
  printf("  #Shells\t= %d\n", CInt_getNumShells(basis));
  printf("  #Funcs\t= %d\n", CInt_getNumFuncs(basis));
  printf("  #OccOrb\t= %d\n", CInt_getNumOccOrb(basis));

  int n = CInt_getNumFuncs(basis);
  int n2 = n * n;
  int n3 = n2 * n;
  int n4 = n3 * n;
  
  int rank = 0;

  ERD_t erd;
  CInt_createERD(basis, &erd, 1);

  printf("Computing Lazy Evaluation Cholesky of ERIs\n");
  double* G_lazyERI;
  //= (double*) calloc(n2*max_num_cols,sizeof(double));
  cholERI(basis, erd, &G_lazyERI, tol, &rank);
  
  printf("Diagonal from Cholesky\n");
  for (int i = 0; i < n2; i++) {
    double aii = 0;
    for (int j = 0; j < rank; j++) {
      //    printf("G[%d,%d]=%1.2f, ",i,j,G_lazyERI[i+j*n2]);
      aii += G_lazyERI[i+j*n2]*G_lazyERI[i+j*n2];
    }
    printf("A[%d,%d]=%f\n",i,i,aii);
  }
  
  printf("rank: %d\n",rank);
  int nshell = CInt_getNumShells(basis);
  int shellIndexM, shellIndexN, shellIndexP, shellIndexQ;
  
  for (shellIndexM = 0; shellIndexM < nshell; shellIndexM++) {
    for (shellIndexN = 0; shellIndexN < nshell; shellIndexN++) {
      for (shellIndexP = 0; shellIndexP < nshell; shellIndexP++) {
	for (shellIndexQ = 0; shellIndexQ < nshell; shellIndexQ++) {
	  printf("Computing a shell quartet from the Cholesky factor\n");	  
	  double *cholintegrals;
	  int cholnints;
	  cholComputeShellQuartet(basis, G_lazyERI, rank, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &cholintegrals, &cholnints);
	  printf("cholnints: %d\n",cholnints);
	  printf("Computing same shell quartet using CInt\n");
	  // Compute the same shell quartet with CInt  
	  double *integrals;
	  int nints;
	  CInt_computeShellQuartet(basis, erd, 0, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &integrals, &nints);
	  printf("nints: %d\n",nints);
	  int dimM = CInt_getShellDim (basis, shellIndexM);
	  int dimN = CInt_getShellDim (basis, shellIndexN);
	  int dimP = CInt_getShellDim (basis, shellIndexP);
	  int dimQ = CInt_getShellDim (basis, shellIndexQ);
	  int startM = CInt_getFuncStartInd (basis, shellIndexM);
	  int startN = CInt_getFuncStartInd (basis, shellIndexN);
	  int startP = CInt_getFuncStartInd (basis, shellIndexP);
	  int startQ = CInt_getFuncStartInd (basis, shellIndexQ);
	  printf("M: %d, N: %d, P: %d, Q: %d\n",shellIndexM,shellIndexN,shellIndexP,shellIndexQ);
	  printf("dimM: %d, dimN: %d, dimP: %d, dimQ: %d\n",
		 dimM, dimN, dimP, dimQ);
	  // Compare the shell quartets from Cholesky and CInt
	  
	  //  for (int i = 0; i < fmin(nints,cholnints); i++){
	  for (int iM = 0; iM < dimM; iM++) {
	    for (int iN = 0; iN < dimN; iN++) {
	      for (int iP = 0; iP < dimP; iP++) {
		for (int iQ = 0; iQ < dimQ; iQ++) {
		  int idx = iM + dimM * (iN + dimN * (iP + dimP *(iQ)));
		  printf("(%d %d | %d %d): Chol = %2.1f, CInt = %2.1f\n",startM+iM,startN+iN,startP+iP,startQ+iQ,cholintegrals[idx],integrals[idx]);
		  double abserror = fabs(integrals[idx] - cholintegrals[idx]);
		  printf("abserror: %2.1e, tol: %2.1e\n",abserror,tol);
		  if (abserror > tol) {
		    printf("-> Error!\n");
		  }
		  else {
		    printf("-> Satifies tolerance\n");
		  }
		}
	      }
	    }
	  }
	  free(cholintegrals);
	}
      }
    }
  }
}
