#include "matrices.h"

//prints the elements of a matrix in a file
void printmatrix(const char* filename,gsl_matrix* m)
{
    int i,j;
    double s;
    FILE* out = fopen(filename,"w");
    
    if(NULL==out)
    {
        printf("Cannot open output file [%s]\n",filename);
        exit(1);
    }
    for(i=0;i<m->size1;i++)
    {
        fprintf(out,"%.3lf",gsl_matrix_get(m,i,0));
        for(j=1;j<m->size2;j++)
        {
            fprintf(out,"\t%.3lf", gsl_matrix_get(m,i,j));
        }
        fprintf(out,"\n");
    }
    fclose(out);
    return;
}

//creates the transpose of the matrix m
gsl_matrix* transposematrix(gsl_matrix* m)
{
    int i,j;
    
    gsl_matrix* tm = gsl_matrix_alloc(m->size2,m->size1);
    
    for(i=0;i<tm->size1;i++)
    {
        for(j=0;j<tm->size2;j++)
        {
            gsl_matrix_set(tm,i,j,gsl_matrix_get(m,j,i));
        }
    }
    
    return(tm);
}

//calculates the product of a nxp matrix m1 with a pxl matrix m2
//returns a nxl matrix m
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m)
{
    int i,j,k;
    double s;
    
    for(i=0;i<m->size1;i++)
    {
        for(k=0;k<m->size2;k++)
        {
            s = 0;
            for(j=0;j<m1->size2;j++)
            {
                s += gsl_matrix_get(m1,i,j)*gsl_matrix_get(m2,j,k);
            }
            gsl_matrix_set(m,i,k,s);
        }
    }
    return;
}


//computes the inverse of a positive definite matrix
//the function returns a new matrix which contains the inverse
//the matrix that gets inverted is not modified
void  getInverse(gsl_matrix* K, gsl_matrix* inverse)
{
    int j;
    
    gsl_matrix* copyK = gsl_matrix_alloc(K->size1,K->size1);
    if(GSL_SUCCESS!=gsl_matrix_memcpy(copyK,K))
    {
        printf("GSL failed to copy a matrix.\n");
        exit(1);
    }
    
    gsl_permutation *myperm = gsl_permutation_alloc(K->size1);
    
    if(GSL_SUCCESS!=gsl_linalg_LU_decomp(copyK,myperm,&j))
    {
        printf("GSL failed LU decomposition.\n");
        exit(1);
    }
    if(GSL_SUCCESS!=gsl_linalg_LU_invert(copyK,myperm,inverse))
    {
        printf("GSL failed matrix inversion.\n");
        exit(1);
    }
    gsl_permutation_free(myperm);
    gsl_matrix_free(copyK);
    
    return;
}

//creates a submatrix of matrix M
//the indices of the rows and columns to be selected are
//specified in the last four arguments of this function
gsl_matrix* MakeSubmatrix(gsl_matrix* M,int* IndRow,int lenIndRow,int* IndColumn,int lenIndColumn)
{
    int i,j;
    gsl_matrix* subM = gsl_matrix_alloc(lenIndRow,lenIndColumn);
    
    for(i=0;i<lenIndRow;i++)
    {
        for(j=0;j<lenIndColumn;j++)
        {
            gsl_matrix_set(subM,i,j,
                           gsl_matrix_get(M,IndRow[i],IndColumn[j]));
        }
    }
    
    return(subM);
}

//computes the log of the determinant of a symmetric positive definite matrix
double logdet(gsl_matrix* K)
{
    int i;
    
    gsl_matrix* CopyOfK = gsl_matrix_alloc(K->size1,K->size2);
    gsl_matrix_memcpy(CopyOfK,K);
    gsl_permutation *myperm = gsl_permutation_alloc(K->size1);
    if(GSL_SUCCESS!=gsl_linalg_LU_decomp(CopyOfK,myperm,&i))
    {
        printf("GSL failed LU decomposition.\n");
        exit(1);
    }
    double logdet = gsl_linalg_LU_lndet(CopyOfK);
    gsl_permutation_free(myperm);
    gsl_matrix_free(CopyOfK);
    return(logdet);
}

//create the diagonal matrix of n*n
gsl_matrix* diagMatrix(int n)
{
    int i;
    gsl_matrix* mat = gsl_matrix_alloc(n, n);
    gsl_matrix_set_all(mat, 0.0);
    for (i=0; i<n ;i++)
    {
        gsl_matrix_set(mat, i, i, 1);
    }
    return mat;
}


//cholesky decomposition
gsl_matrix* getcholesky(gsl_matrix* m)
{
    size_t n = m->size1;
    size_t p = m->size2;
    gsl_matrix* m_copy = gsl_matrix_alloc(n, n);
    
    //check whether it's square matrix
    if (n!=p)
    {
        printf("not a square matrix\n");
        exit(1);
    }
    
    if(GSL_SUCCESS!=gsl_matrix_memcpy(m_copy, m))
    {
        printf("GSL failed to copy a matrix.\n");
        exit(1);
    }
    
    //get cholesky decomposition
    gsl_linalg_cholesky_decomp(m_copy);
    
    //transform m_copy to a upper triangle matrix
    for (int i=0; i<n; i++)
    {
        for (int j=i+1; j<n; j++)
        {
            gsl_matrix_set(m_copy, i, j, 0);
        }
    }
    
    return (m_copy);
}


//gsl_matrix* getsimusample(gsl_rng* r, gsl_matrix* mat_chol, int size, int p)
gsl_matrix* getsimusample(int size, int p, gsl_rng* r, gsl_matrix* mat_chol)
{
    int i, j;
    gsl_matrix* Z = gsl_matrix_alloc(p, 1);
    gsl_matrix* X = gsl_matrix_alloc(p, 1);
    gsl_matrix* data_simulation = gsl_matrix_alloc(size, p);
    
    //randomly draw samples
    for (i = 0; i<size; i++)
    {
        for (j = 0; j<p; j++)
        {
            gsl_matrix_set(Z, j, 0, gsl_ran_ugaussian(r));
        }
        
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mat_chol, Z, 0.0, X);
        
        for (j = 0; j<p; j++)
        {
            gsl_matrix_set(data_simulation, i, j, gsl_matrix_get(X, j, 0));
        }
    }
    
    gsl_matrix_free(Z);
    gsl_matrix_free(X);
    
    return (data_simulation);
}


//compute the inverse of logit
gsl_vector*  getInverseLogit(gsl_vector*  x)
{
    size_t n = x->size;
    gsl_vector* inverseLogit = gsl_vector_alloc(n);
    for(int i = 0; i < n; i++)
    {
        gsl_vector_set(inverseLogit, i, exp(gsl_vector_get(x,i)) / (1 + exp(gsl_vector_get(x, i))));
    }
    return(inverseLogit);
}

//compute the inverse of logit2
gsl_vector* getInverseLogit2(gsl_vector* x)
{
    size_t n = x->size;
    gsl_vector* inverseLogit2 = gsl_vector_alloc(n);
    for(int i = 0; i < n; i++)
    {
        gsl_vector_set(inverseLogit2, i, exp(gsl_vector_get(x, i)) / pow((1 + exp(gsl_vector_get(x, i))),2));
    }
    return(inverseLogit2);
}

//compute P(yi=1|xi)
gsl_vector* getPi(gsl_vector* x, gsl_vector* beta)
{
    size_t n = x->size;
    gsl_matrix* matX = gsl_matrix_alloc(n, 2);
    // betaX is beta0 + beta1*x
    gsl_vector* betaX = gsl_vector_alloc(n);
    
    for(int i = 0; i < n; i++)
    {
        gsl_matrix_set(matX, i, 0, 1);
        gsl_matrix_set(matX, i, 1, gsl_vector_get(x, i));
    }
    
    for(int i = 0; i < n; i++)
    {
        // compute beta0 + beta1*x
        gsl_vector_set(betaX, i, gsl_matrix_get(matX, i, 0)*gsl_vector_get(beta, 0) + gsl_matrix_get(matX, i, 1)*gsl_vector_get(beta, 1));
    }
    gsl_vector* Pi = getInverseLogit(betaX);
    
    gsl_matrix_free(matX);
    gsl_vector_free(betaX);
    return(Pi);
}


//compute Pi2, which is useful in computing Hessian matrix
gsl_vector* getPi2(gsl_vector* x, gsl_vector* beta)
{
    size_t n = x->size;
    gsl_matrix* matX = gsl_matrix_alloc(n, 2);
    //betaX is beta0 + beta1*x
    gsl_vector* betaX = gsl_vector_alloc(n);
    
    for(int i = 0; i < n; i++)
    {
        gsl_matrix_set(matX, i, 0, 1);
        gsl_matrix_set(matX, i, 1, gsl_vector_get(x, i));
    }
    
    for(int i = 0; i < n; i++)
    {
        // compute beta0 + beta1*x
        gsl_vector_set(betaX, i, gsl_matrix_get(matX, i, 0)*gsl_vector_get(beta, 0)+ gsl_matrix_get(matX, i, 1)*gsl_vector_get(beta, 1));
    }
    gsl_vector* Pi2 = getInverseLogit2(betaX);
    
    gsl_matrix_free(matX);
    gsl_vector_free(betaX);
    return(Pi2);
}



//compute logistic loglikelihood
double getLoglik(gsl_vector* y, gsl_vector* x, gsl_vector* beta)
{
    size_t n = x->size;
    double loglik = 0.0;
    gsl_vector* Pi = getPi(x, beta);
    
    for(int i = 0; i < n; i++)
    {
        loglik += gsl_vector_get(y, i)*log(gsl_vector_get(Pi, i)) + (1 - gsl_vector_get(y, i))*log(1 - gsl_vector_get(Pi, i));
    }
    
    gsl_vector_free(Pi);
    return(loglik);
}



//compute Lstar. Lstar = -0.5*(beta0^2+beta1^2)+loglik. We omit -log(2*pi) because it would be cancelled out in logmarglik
double getLstar(gsl_vector* y, gsl_vector* x, gsl_vector* beta)
{
    double betasquaresum = -0.5*(pow(gsl_vector_get(beta, 0),2)+ pow(gsl_vector_get(beta, 1), 2));
    double loglik = getLoglik(y, x, beta);
    double Lstar = betasquaresum + loglik;
    return(Lstar);
}


//compute gradient
void getGradient(gsl_vector* y, gsl_vector* x, gsl_vector* beta, gsl_matrix* Gradient)
{
    size_t n = x->size;
    double element1 = 0.0;
    double element2 = 0.0;
     gsl_vector* Pi = getPi(x, beta);
    
    for (int i = 0; i < n; i++)
    {
        element1 += gsl_vector_get(y, i) - gsl_vector_get(Pi, i);
        element2 += (gsl_vector_get(y, i) - gsl_vector_get(Pi, i))*gsl_vector_get(x, i);
    }
    
    gsl_matrix_set(Gradient, 0, 0, element1 - gsl_vector_get(beta, 0));
    gsl_matrix_set(Gradient, 1, 0, element2 - gsl_vector_get(beta, 1));
    
    gsl_vector_free(Pi);
    return;
}


//compute Hessian Matrix
void getHessian(gsl_vector* y, gsl_vector* x, gsl_vector* beta, gsl_matrix* Hessian)
{
    size_t n = x->size;
    double element1 = 0.0;
    double element2 = 0.0;
    double element3 = 0.0;
    gsl_vector* Pi2 = getPi2(x, beta);
    
    for (int i=0; i<n; i++)
    {
        element1 += gsl_vector_get(Pi2, i);
        element2 += gsl_vector_get(Pi2, i)*gsl_vector_get(x, i);
        element3 += gsl_vector_get(Pi2, i)*pow(gsl_vector_get(x, i), 2);
    }
    
    gsl_matrix_set(Hessian, 0, 0, element1 + 1);
    gsl_matrix_set(Hessian, 0, 1, element2);
    gsl_matrix_set(Hessian, 1, 0, element2);
    gsl_matrix_set(Hessian, 1, 1, element3 + 1);
    
    gsl_matrix_scale(Hessian, -1.0);
    
    gsl_vector_free(Pi2);
    return;
}


//find the maxmimum elements in the vector |betaNew - betaCurrent|
double getMaxBetaDiff(gsl_vector* betaCurrent,  gsl_vector* betaNew)
{
    double maxBetaDiff;
    
    maxBetaDiff = gsl_vector_get(betaNew, 0) - gsl_vector_get(betaCurrent, 0);
    if (maxBetaDiff < gsl_vector_get(betaCurrent, 0)-gsl_vector_get(betaNew, 0))
    {
        maxBetaDiff = gsl_vector_get(betaCurrent, 0)-gsl_vector_get(betaNew, 0);
    }
    if (maxBetaDiff < gsl_vector_get(betaNew, 1)-gsl_vector_get(betaCurrent, 1))
    {
        maxBetaDiff = gsl_vector_get(betaNew, 1)-gsl_vector_get(betaCurrent, 1);
    }
    if (maxBetaDiff < gsl_vector_get(betaCurrent, 1)-gsl_vector_get(betaNew, 1))
    {
        maxBetaDiff = gsl_vector_get(betaCurrent, 1)-gsl_vector_get(betaNew, 1);
    }
    
    return(maxBetaDiff);
}


//compute coefficient estimation by using Newton-Raphson procedure
gsl_vector* getcoefNR(int response, int explanatory, gsl_matrix* data, double error)
{
    size_t n = data->size1;
    gsl_vector* betaCurrent = gsl_vector_alloc(2);
    gsl_vector_set(betaCurrent, 0, 0.0);
    gsl_vector_set(betaCurrent, 1, 0.0);
    gsl_vector* betaNew = gsl_vector_alloc(2);
    gsl_vector* x = gsl_vector_alloc(n);
    gsl_vector* y = gsl_vector_alloc(n);
    gsl_matrix* Hessian = gsl_matrix_alloc(2, 2);
    gsl_matrix* Gradient = gsl_matrix_alloc(2, 1);
    gsl_matrix* InvHessian = gsl_matrix_alloc(2, 2);
    gsl_matrix* ExtraTerm = gsl_matrix_alloc(2, 1);
    double maxBetaDiff;
    double newLstar;
    double currentLstar;
    
    //get response and explanatory varaibles
    for (int i = 0; i < n; i++)
    {
        gsl_vector_set(y, i, gsl_matrix_get(data, i, response));
        gsl_vector_set(x, i, gsl_matrix_get(data, i, explanatory));
    }
    
    //initialize Lstar
    currentLstar = getLstar(y, x, betaCurrent);
    
    //loop until we find a good enough beta estimation
    while (1)
    {
        getHessian(y, x, betaCurrent, Hessian);
        getGradient(y, x, betaCurrent, Gradient);
        getInverse(Hessian,InvHessian);
        matrixproduct(InvHessian, Gradient, ExtraTerm);
        
        //betaNew = betaCurrent + ExtraTerm
        gsl_vector_set(betaNew, 0, gsl_vector_get(betaCurrent, 0) - gsl_matrix_get(ExtraTerm, 0, 0));
        gsl_vector_set(betaNew, 1, gsl_vector_get(betaCurrent, 1) - gsl_matrix_get(ExtraTerm, 1, 0));
        
        //use betaNew to get newLstar
        newLstar = getLstar(y, x, betaNew);
        
        //stop if Lstar doesn't increase (greedy algorithrmn)
        if (newLstar < currentLstar)
        {
            break;
        }
        
        //find the maxmimum elements in the vector |betaNew - betaCurrent|
        //stop if |betaNew - betaCurrent|<1e-6
        maxBetaDiff = getMaxBetaDiff(betaCurrent, betaNew);
        
        if (maxBetaDiff<error)
        {
            break;
        }
        
        gsl_vector_set(betaCurrent, 0, gsl_vector_get(betaNew, 0));
        gsl_vector_set(betaCurrent, 1, gsl_vector_get(betaNew, 1));
        currentLstar = newLstar;
    }
    
    gsl_vector_free(betaNew);
    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_matrix_free(Hessian);
    gsl_matrix_free(Gradient);
    gsl_matrix_free(InvHessian);
    gsl_matrix_free(ExtraTerm);
    return(betaCurrent);
}


//compute the log marginal likelihood by using the Laplace approximation
double getLaplace(int response, int explanatory, gsl_matrix* data, gsl_vector* betaMode)
{
    size_t n = data->size1;
    gsl_vector* y = gsl_vector_alloc(n);
    gsl_vector* x = gsl_vector_alloc(n);
    gsl_matrix* Hessian = gsl_matrix_alloc(2, 2);
    
    //get response and explanatory varaibles
    for (int i=0; i<n; i++)
    {
        gsl_vector_set(y, i, gsl_matrix_get(data, i, response));
        gsl_vector_set(x, i, gsl_matrix_get(data, i, explanatory));
    }
    
    double betasquaresum = -0.5*(pow(gsl_vector_get(betaMode,0),2)+pow(gsl_vector_get(betaMode,1),2));
    double loglik = getLoglik(y, x, betaMode);
    //get the negative inverse Hessian matrix where beta = betaMode
    getHessian(y, x, betaMode, Hessian);
    gsl_matrix_scale(Hessian, -1.0);
    
    double logmarglik =  betasquaresum + loglik - 0.5*logdet(Hessian);
    
    gsl_vector_free(y);
    gsl_vector_free(x);
    gsl_matrix_free(Hessian);
    return(logmarglik);
}


//compute the log marginal likelihood by using Monte Carlo simulation
double getMonteCarlo(int NumIterations, gsl_rng* r, int response, int explanatory, gsl_matrix* data)
{
    size_t n = data->size1;
    gsl_vector* y = gsl_vector_alloc(n);
    gsl_vector* x = gsl_vector_alloc(n);
    gsl_vector* loglikSimulation = gsl_vector_alloc(NumIterations);
    gsl_vector* loglikSimulationCopy = gsl_vector_alloc(NumIterations);
    
    //get response and explanatory varaibles
    for (int i=0; i<n; i++)
    {
        gsl_vector_set(y, i, gsl_matrix_get(data, i, response));
        gsl_vector_set(x, i, gsl_matrix_get(data, i, explanatory));
    }
    
    //simulate beta
    for (int i=0; i<NumIterations; i++)
    {
        gsl_vector* betaSimulation = gsl_vector_alloc(2);
        gsl_vector_set(betaSimulation, 0, gsl_ran_ugaussian(r));
        gsl_vector_set(betaSimulation, 1, gsl_ran_ugaussian(r));
        gsl_vector_set(loglikSimulation, i, getLoglik(y, x, betaSimulation));
        
        gsl_vector_free(betaSimulation);
    }
    
    //get the maximum value of the vector loglikSimulation
    double maxloglikSimulation = gsl_vector_max(loglikSimulation);
    
    for (int i=0; i<NumIterations; i++)
    {
        gsl_vector_set(loglikSimulationCopy, i, exp(gsl_vector_get(loglikSimulation,i)- maxloglikSimulation));
    }
    
    double temp = log(gsl_stats_mean(loglikSimulationCopy->data, loglikSimulationCopy->stride, NumIterations));
    double logmarglik = temp + maxloglikSimulation;
    
    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(loglikSimulation);
    gsl_vector_free(loglikSimulationCopy);
    return(logmarglik);
}


//Use Metropolis-Hastings algorithm to update beta estimation
gsl_vector* getMHinteration(gsl_rng* r, gsl_vector* y, gsl_vector* x, gsl_vector* betaCurrent, gsl_matrix* NegInvHessian)
{
    gsl_matrix* cholesky = getcholesky(NegInvHessian);
    gsl_matrix* randomsample = getsimusample(1, 2, r, cholesky);
    gsl_vector* betaNew = gsl_vector_alloc(2);
    gsl_vector_set(betaNew, 0, gsl_vector_get(betaCurrent, 0) + gsl_matrix_get(randomsample, 0, 0));
    gsl_vector_set(betaNew, 1, gsl_vector_get(betaCurrent, 1) + gsl_matrix_get(randomsample, 0, 1));
    
    double currentLStar = getLstar(y, x, betaCurrent);
    double newLStar = getLstar(y, x, betaNew);
    double randomdraw = gsl_rng_uniform(r);
    
    gsl_matrix_free(cholesky);
    gsl_matrix_free(randomsample);
    
    if (currentLStar <= newLStar)
    {
        //accept and return the new beta
        return(betaNew);
    }
    
    if (randomdraw <= exp(newLStar - currentLStar))
    {
        //accept and return the new beta
        return(betaNew);
    }
    
    //reject and return the old beta
    gsl_vector_free(betaNew);
    return(betaCurrent);
    
    
}


//compute posterior coefficient by using Metropolis-Hastings algorithm
gsl_vector* getcoefMH(int NumIterations, gsl_rng* r, int response, int explanatory, gsl_matrix* data, gsl_vector* betaMode)
{
    size_t n = data->size1;
    gsl_vector* y = gsl_vector_alloc(n);
    gsl_vector* x = gsl_vector_alloc(n);
    gsl_matrix* Hessian = gsl_matrix_alloc(2, 2);
    gsl_matrix* NegInvHessian = gsl_matrix_alloc(2, 2);
    gsl_vector* betaMH = gsl_vector_alloc(2);
    gsl_vector_set(betaMH, 0, 0);
    gsl_vector_set(betaMH, 1, 0);
    
    //set initial beta
    gsl_vector* betaCurrent = betaMode;
    
    //get response and explanatory varaibles
    for (int i=0; i<n; i++)
    {
        gsl_vector_set(y, i, gsl_matrix_get(data, i, response));
        gsl_vector_set(x, i, gsl_matrix_get(data, i, explanatory));
    }
    
    //get the negative inverse Hessian matrix where beta = betaMode
    getHessian(y, x, betaMode, Hessian);
    getInverse(Hessian, NegInvHessian);
    gsl_matrix_scale(NegInvHessian, -1.0);
    
    for (int i=0; i<NumIterations; i++)
    {
        betaCurrent = getMHinteration(r, y, x, betaCurrent, NegInvHessian);
        gsl_vector_add(betaMH, betaCurrent);
    }
    
    gsl_vector_scale(betaMH, 1.0/NumIterations);
    
    gsl_vector_free(y);
    gsl_vector_free(x);
    gsl_matrix_free(Hessian);
    gsl_matrix_free(NegInvHessian);
    gsl_vector_free(betaCurrent);
    return(betaMH);
}

