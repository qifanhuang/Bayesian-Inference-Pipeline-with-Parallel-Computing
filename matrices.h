#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

//THESE ARE GSL FUNCTIONS
//YOU DO NOT NEED TO INCLUDE ALL THESE HEADER FILES IN YOUR CODE
//JUST THE ONES YOU ACTUALLY NEED;
//IN THIS APPLICATION, WE ONLY NEED gsl/gsl_matrix.h
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_errno.h>

void printmatrix(const char* filename,gsl_matrix* m);
gsl_matrix* transposematrix(gsl_matrix* m);
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m);
void getInverse(gsl_matrix* K, gsl_matrix* inverse);
gsl_matrix* MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn);
double logdet(gsl_matrix* K);
gsl_matrix* diagMatrix(int n);
gsl_matrix* getcholesky(gsl_matrix* m);
gsl_matrix* getsimusample(int size, int p, gsl_rng* r, gsl_matrix* mat_chol);
gsl_vector*  getInverseLogit(gsl_vector*  x);
gsl_vector* getInverseLogit2(gsl_vector* x);
gsl_vector* getPi(gsl_vector* x, gsl_vector* beta);
gsl_vector* getPi2(gsl_vector* x, gsl_vector* beta);
double getLoglik(gsl_vector* y, gsl_vector* x, gsl_vector* beta);
double getLstar(gsl_vector* y, gsl_vector* x, gsl_vector* beta);
void getGradient(gsl_vector* y, gsl_vector* x, gsl_vector* beta, gsl_matrix* Gradient);
void getHessian(gsl_vector* y, gsl_vector* x, gsl_vector* beta, gsl_matrix* Hessian);
double getMaxBetaDiff(gsl_vector* betaCurrent,  gsl_vector* betaNew);
gsl_vector* getcoefNR(int response, int explanatory, gsl_matrix* data, double error);
gsl_vector* getMHinteration(gsl_rng* r, gsl_vector* y, gsl_vector* x, gsl_vector* betaCurrent, gsl_matrix* NegInvHessian);
double getLaplace(int response, int explanatory, gsl_matrix* data, gsl_vector* betaMode);
double getMonteCarlo(int NumIterations, gsl_rng* r, int response, int explanatory, gsl_matrix* data);
gsl_vector* getcoefMH(int NumIterations, gsl_rng* r, int response, int explanatory, gsl_matrix* data, gsl_vector* betaMode);

