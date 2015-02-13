#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <sys/time.h>
//#include <config.h>
#include <inttypes.h>

//#include <flops.h>
//#include <ratios.h>
//#include <screening.h>
//#include <erd_profile.h>

#include "cholERI.h"
#include "cholesky.h"
#include "CInt.h"

static uint64_t get_cpu_frequency(void) {
  const uint64_t start_clock = __rdtsc();
  sleep(1);
  const uint64_t end_clock = __rdtsc();
  return end_clock - start_clock;
}

int main (int argc, char **argv)
{
  if (argc != 4) {
    printf ("Usage: %s <basisset> <xyz>\n", argv[0]);
    return -1;
  }
  
  const uint64_t freq = get_cpu_frequency();
  const int nthreads = atoi(argv[3]);
  /*
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#else
  assert(nthreads == 1);
#endif
  */
  // load basis set
  BasisSet_t basis;
  CInt_createBasisSet(&basis);
  CInt_loadBasisSet(basis, argv[1], argv[2]);
  
  printf("Molecule info:\n");
  printf("  #Atoms\t= %d\n", CInt_getNumAtoms(basis));
  printf("  #Shells\t= %d\n", CInt_getNumShells(basis));
  printf("  #Funcs\t= %d\n", CInt_getNumFuncs(basis));
  printf("  #OccOrb\t= %d\n", CInt_getNumOccOrb(basis));

  ERD_t erd;
  CInt_createERD(basis, &erd, nthreads);

  printf("Computing Lazy Evaluation Cholesky of ERIs\n");
  // reset profiler
  //  erd_reset_profile();

  int n = CInt_getNumFuncs(basis);
  int n2 = n * n;
  int n3 = n2 * n;
  int n4 = n3 * n;
  
  double* G_ERI;
  double tol = 1e-6;
  int max_rank = n2;
  //int max_rank = (1-floor(log10(tol)))*n;
  int rank;
  const uint64_t start_clock = __rdtsc();
  cholERI(basis, erd, &G_ERI, tol, max_rank, &rank);
  const uint64_t end_clock = __rdtsc();
  const uint64_t total_ticks = end_clock - start_clock;
  const double timepass = ((double) total_ticks) / freq;
  printf("Done\n");
  printf("Total GigaTicks: %.3lf, freq = %.3lf GHz\n", (double) (total_ticks) * 1.0e-9, (double)freq/1.0e9);
  printf("Total time: %.4lf secs\n", timepass);

  printf("n: %d, rank: %d, 7n: %d, n2: %d\n",n,rank,7*n,n2);
  double* diag = (double*) malloc(n2*sizeof(double));
  computeDiag(basis, erd, diag);
  for (int i = 0; i < n2; i++) {
    double aii = 0;
    for (int j = 0; j < rank; j++) {
      aii += G_ERI[i+j*n2] * G_ERI[i+j*n2];
    }
    double abserror2 = (diag[i] - aii)*(diag[i] - aii);
    if (abserror2 > tol)
      printf("i=%d, truth=%1.2e, approx=%1.2e, error: %1.2e\n", i, diag[i], aii, abserror2);
  }
  free(diag);

  printf("Testing accuracy for each shell quartet\n");
  double chol_time_total = 0;
  double CInt_time_total = 0;
  int nshell = CInt_getNumShells(basis);
  int shellIndexM, shellIndexN, shellIndexP, shellIndexQ;
  int correct = 1;
  for (shellIndexM = 0; shellIndexM < nshell; shellIndexM++) {
    for (shellIndexN = 0; shellIndexN < nshell; shellIndexN++) {
      for (shellIndexP = 0; shellIndexP < nshell; shellIndexP++) {
	for (shellIndexQ = 0; shellIndexQ < nshell; shellIndexQ++) {
	  int dimM = CInt_getShellDim (basis, shellIndexM);
	  int dimN = CInt_getShellDim (basis, shellIndexN);
	  int dimP = CInt_getShellDim (basis, shellIndexP);
	  int dimQ = CInt_getShellDim (basis, shellIndexQ);

	  // Compute shell with Cholesky
	  double *cholintegrals;
	  int cholnints;
	  const uint64_t chol_start = __rdtsc();
	  cholComputeShellQuartet(basis, G_ERI, rank, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &cholintegrals, &cholnints);
	  const uint64_t chol_end = __rdtsc();
	  chol_time_total += ((double) chol_end - chol_start) / freq;
	  
	  // Compute the same shell quartet with CInt
	  double *integrals;
	  int nints;
	  const uint64_t CInt_start = __rdtsc();
	  CInt_computeShellQuartet(basis, erd, 0, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &integrals, &nints);
	  const uint64_t CInt_end = __rdtsc();
	  CInt_time_total += ((double) CInt_end - CInt_start) / freq;
	  
	  // Compare each integral individually
	  for (int iM = 0; iM < dimM; iM++) {
	    for (int iN = 0; iN < dimN; iN++) {
	      for (int iP = 0; iP < dimP; iP++) {
		for (int iQ = 0; iQ < dimQ; iQ++) {
		  int idx = iM + dimM * (iN + dimN * (iP + dimP *(iQ)));
		  double abserror2 = (integrals[idx] - cholintegrals[idx]);
		  abserror2 = abserror2*abserror2;
		  if (abserror2 > tol) {
		    correct = 0;
		    printf("Integral does not satisfy error tolerance: error = %1.2e\n",abserror2);
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
  printf("Total time to eval shells from Cholesky factor: %.4lf secs\n", chol_time_total);
  printf("Total time to eval shells with CInt: %.4lf secs\n", CInt_time_total);
  printf("Total time to compute Cholesky factor and eval shells from Cholesky factor: %.4lf secs\n", timepass+chol_time_total);

  printf("Computing Structured Lazy Evaluation Cholesky of ERIs\n");
  double* G_structERI;
  max_rank = (n*(n+1))/2;
  const uint64_t struct_start_clock = __rdtsc();
  structcholERI(basis, erd, &G_structERI, tol, max_rank, &rank);
  const uint64_t struct_end_clock = __rdtsc();
  const uint64_t struct_total_ticks = struct_end_clock - struct_start_clock;
  const double struct_timepass = ((double) struct_total_ticks) / freq;
  printf("Done\n");
  printf("Total GigaTicks: %.3lf, freq = %.3lf GHz\n", (double) (struct_total_ticks) * 1.0e-9, (double)freq/1.0e9);
  printf("Total time: %.4lf secs\n", struct_timepass);
  
  
  double* structdiag = (double*) malloc(max_rank*sizeof(double));
  computeStructDiag(basis, erd, structdiag);
  for (int i = 0; i < max_rank; i++) {
    double aii = 0;
    for (int j = 0; j < rank; j++) {
      aii += G_structERI[i+j*max_rank] * G_structERI[i+j*max_rank];
    }
    double abserror2 = (structdiag[i] - aii)*(structdiag[i] - aii);
    if (abserror2 > tol)
      printf("i=%d, truth=%1.2e, approx=%1.2e, error: %1.2e\n", i, structdiag[i], aii, abserror2);
  }
  free(structdiag);
  printf("Testing accuracy for each shell quartet\n");
  double struct_chol_time_total = 0;
  CInt_time_total = 0;
  correct = 1;
  for (shellIndexM = 0; shellIndexM < nshell; shellIndexM++) {
    for (shellIndexN = 0; shellIndexN < nshell; shellIndexN++) {
      for (shellIndexP = 0; shellIndexP < nshell; shellIndexP++) {
	for (shellIndexQ = 0; shellIndexQ < nshell; shellIndexQ++) {
	  int dimM = CInt_getShellDim (basis, shellIndexM);
	  int dimN = CInt_getShellDim (basis, shellIndexN);
	  int dimP = CInt_getShellDim (basis, shellIndexP);
	  int dimQ = CInt_getShellDim (basis, shellIndexQ);

	  // Compute shell with structured Cholesky
	  double *cholintegrals;
	  int cholnints;
	  const uint64_t struct_chol_start = __rdtsc();
	  structcholComputeShellQuartet(basis, G_structERI, rank, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &cholintegrals, &cholnints);
	  const uint64_t struct_chol_end = __rdtsc();
	  struct_chol_time_total += ((double) struct_chol_end - struct_chol_start) / freq;
	  
	  // Compute the same shell quartet with CInt  
	  double *integrals;
	  int nints;
	  const uint64_t CInt_start = __rdtsc();
	  CInt_computeShellQuartet(basis, erd, 0, shellIndexM, shellIndexN, shellIndexP, shellIndexQ, &integrals, &nints);
	  const uint64_t CInt_end = __rdtsc();
	  CInt_time_total += ((double) CInt_end - CInt_start) / freq;
	  
	  // Compare each integral individually
	  for (int iM = 0; iM < dimM; iM++) {
	    for (int iN = 0; iN < dimN; iN++) {
	      for (int iP = 0; iP < dimP; iP++) {
		for (int iQ = 0; iQ < dimQ; iQ++) {
		  int idx = iM + dimM * (iN + dimN * (iP + dimP *(iQ)));
		  double abserror2 = (integrals[idx] - cholintegrals[idx]);
		  abserror2 = abserror2*abserror2;
		  if (abserror2 > tol) {
		    correct = 0;
		    printf("Integral does not satisfy error tolerance: error = %1.2e\n",abserror2);
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
  printf("Total time to eval shells from structured Cholesky factor: %.4lf secs\n", struct_chol_time_total);
  printf("Total time to eval shells with CInt: %.4lf secs\n", CInt_time_total);
  printf("Total time to compute Cholesky factor and eval shells from Cholesky factor: %.4lf secs\n", struct_timepass+struct_chol_time_total);

  free(G_structERI);  
  free(G_ERI);
  return 0;
}
