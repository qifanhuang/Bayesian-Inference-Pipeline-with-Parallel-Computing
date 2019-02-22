#ifndef _REGMODELS
#define _REGMODELS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct myRegression* LPRegression;
typedef struct myRegression Regression;

struct myRegression
{
  int numRegressor;     //number of regressors
  double logmarglikMon; //Laplace approximation
  double logmarglikLap; //Monte Carlo integration
  double beta0;         //intercept
  double beta1;         //slope
  int regressor;        //regressors

  LPRegression Next;    //link to the next regression
};

int IsSameRegression(int lenReg1,int lenReg2,int reg1,int reg2);
void AddRegression(int nMaxRegs, LPRegression regressions,int numRegressor,int regressor,double logmarglikMon, double logmarglikLap, double beta0, double beta1);
void DeleteAllRegressions(LPRegression regressions);
void DeleteLastRegression(LPRegression regressions);
void SaveRegressions(char* filename,LPRegression regressions);

#endif
