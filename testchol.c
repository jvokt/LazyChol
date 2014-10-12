#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cholERI.h"
#include "cholesky.h"
#include "CInt.h"

int main (int argc, char **argv)
{
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
  
  ERD_t erd;
  CInt_createERD(basis, &erd, 1);

  printf("Computing Lazy Evaluation Cholesky of ERIs\n");
  double* G_lazyERI;
  double tol = 1e-6;
  int max_rank = n2;//(1-floor(log10(tol)))*n;
  int rank;
  cholERI(basis, erd, &G_lazyERI, tol, max_rank, &rank);
  printf("Done. Testing accuracy for each shell quartet\n");
  int nshell = CInt_getNumShells(basis);
  int shellIndexM, shellIndexN, shellIndexP, shellIndexQ;
  int correct = 1;
  for (shellIndexM = 0; shellIndexM < nshell; shellIndexM++) {
    for (shellIndexN = 0; shellIndexN < nshell; shellIndexN++) {
      for (shellIndexP = 0; shellIndexP < nshell; shellIndexP++) {
	for (shellIndexQ = 0; shellIndexQ < nshell; shellIndexQ++) {
	  printf("Computing shell quartet ( %d %d | %d %d )\n",shellIndexM,shellIndexN,shellIndexP,shellIndexQ);
	  int dimM = CInt_getShellDim (basis, shellIndexM);
	  int dimN = CInt_getShellDim (basis, shellIndexN);
	  int dimP = CInt_getShellDim (basis, shellIndexP);
	  int dimQ = CInt_getShellDim (basis, shellIndexQ);

	  // Compute shell with Cholesky
	  double *cholintegrals;
	  int cholnints;
	  cholComputeShellQuartet(basis, G_lazyERI, rank, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &cholintegrals, &cholnints);

	  // Compute the same shell quartet with CInt  
	  double *integrals;
	  int nints;
	  CInt_computeShellQuartet(basis, erd, 0, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &integrals, &nints);
	  
	  // Compare each integral individually
	  for (int iM = 0; iM < dimM; iM++) {
	    for (int iN = 0; iN < dimN; iN++) {
	      for (int iP = 0; iP < dimP; iP++) {
		for (int iQ = 0; iQ < dimQ; iQ++) {
		  int idx = iM + dimM * (iN + dimN * (iP + dimP *(iQ)));
		  double abserror = fabs(integrals[idx] - cholintegrals[idx]);
		  if (abserror > tol) {
		    correct = 0;
		    printf("-> integral does not satisfy error tolerance: error = %1.2f\n",abserror);
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
  if (correct) {
    printf("All integrals in all shell quartets satisfy error tolerance\n");
  } else {
    printf("Some integrals did not satisfy error tolerance\n");
  }
  //  free(G_lazyERI);
  
  printf("Computing Structured Lazy Evaluation Cholesky of ERIs\n");
  double* G;
  max_rank = (n*(n+1))/2;
  structcholERI(basis, erd, &G, tol, max_rank, &rank);
  printf("Done. Testing accuracy for each shell quartet\n");
  correct = 1;
  for (shellIndexM = 0; shellIndexM < nshell; shellIndexM++) {
    for (shellIndexN = 0; shellIndexN < nshell; shellIndexN++) {
      for (shellIndexP = 0; shellIndexP < nshell; shellIndexP++) {
	for (shellIndexQ = 0; shellIndexQ < nshell; shellIndexQ++) {
	  printf("Computing shell quartet ( %d %d | %d %d )\n",shellIndexM,shellIndexN,shellIndexP,shellIndexQ);
	  int dimM = CInt_getShellDim (basis, shellIndexM);
	  int dimN = CInt_getShellDim (basis, shellIndexN);
	  int dimP = CInt_getShellDim (basis, shellIndexP);
	  int dimQ = CInt_getShellDim (basis, shellIndexQ);

	  // Compute shell with structured Cholesky
	  double *cholintegrals;
	  int cholnints;
	  structcholComputeShellQuartet(basis, G, rank, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &cholintegrals, &cholnints);
	  
	  // Compute the same shell quartet with CInt  
	  double *integrals;
	  int nints;
	  CInt_computeShellQuartet(basis, erd, 0, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &integrals, &nints);
	  
	  // Compare each integral individually
	  for (int iM = 0; iM < dimM; iM++) {
	    for (int iN = 0; iN < dimN; iN++) {
	      for (int iP = 0; iP < dimP; iP++) {
		for (int iQ = 0; iQ < dimQ; iQ++) {
		  int idx = iM + dimM * (iN + dimN * (iP + dimP *(iQ)));
		  double abserror = fabs(integrals[idx] - cholintegrals[idx]);
		  if (abserror > tol) {
		    correct = 0;
		    printf("-> integral does not satisfy error tolerance: error = %1.2f\n",abserror);
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
  if (correct) {
    printf("All integrals in all shell quartets satisfy error tolerance\n");
  } else {
    printf("Some integrals did not satisfy error tolerance\n");
  }
  //  free(G_lazyERI);
  
  return 0;
}
