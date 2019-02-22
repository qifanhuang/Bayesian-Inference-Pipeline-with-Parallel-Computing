/*
 FILE: REGMODELS.CPP
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "regmodels.h"



//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPRegression regressions)
{
    //this is the element before the first regression
    LPRegression pprev = regressions;
    //this is the first regression
    LPRegression p = regressions->Next;
    
    //if the list does not have any elements, return
    if(NULL==p)
    {
        return;
    }
    
    //the last element of the list is the only
    //element that has the "Next" field equal to NULL
    while(NULL!=p->Next)
    {
        pprev = p;
        p = p->Next;
    }
    
    //now "p" should give the last element
    p->Next = NULL;
    delete p;
    
    //now the previous element in the list
    //becomes the last element
    pprev->Next = NULL;
    
    return;
}


//this function deletes all the elements of the list
//with the head "regressions"
//remark that the head is not touched
void DeleteAllRegressions(LPRegression regressions)
{
    //this is the first regression
    LPRegression p = regressions->Next;
    LPRegression pnext;
    
    while(NULL!=p)
    {
        //save the link to the next element of p
        pnext = p->Next;
        
        //delete the element specified by p
        p->Next = NULL;
        delete p;
        
        //move to the next element
        p = pnext;
    }
    
    return;
}


//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
void SaveRegressions(char* filename,LPRegression regressions)
{
    //open the output file
    FILE* out = fopen(filename,"w");
    
    if(NULL==out)
    {
        printf("Cannot open output file [%s]\n",filename);
        exit(1);
    }
    
    //this is the first regression
    LPRegression p = regressions->Next;
    
    while(NULL!=p)
    {
        
        //print regressor
        for(int i=0;i<p->numRegressor;i++)
        {
            fprintf(out,"Explanatory variable:\t%d",p->regressor);
        }
        fprintf(out,"\n");
        
        //print log marginal likelhood and coefficient estimation
        fprintf(out,"Laplace approximation of logmarglik:\t %.5lf,\t Monte Carlo integration of logmarglik:\t %.5lf,\t intercept: \t %.5lf, \t slope: \t %.5lf",p->logmarglikLap, p->logmarglikMon, p->beta0, p->beta1);
        
        fprintf(out,"\n");
        
        //go to the next regression
        p = p->Next;
    }
    
    //close the output file
    fclose(out);
    
    return;
}


//test whether two regressions (explanatory variables) are equal
int IsSameRegression(int lenReg1,int lenReg2,int reg1,int reg2)
{
    //same length?
    if(lenReg1!=lenReg2)
    {
        return 0;
    }
    
    //Given same length, are they have same elements?
    if (reg1!=reg2)
    {
        return 0;
    }
    
    return 1;
}

//this function adds a new regression with predictors A
//to the list of regressions. Here "regressions" represents
//the head of the list, "lenA" is the number of predictors
//and "logmarglikA" is the marginal likelihood of the regression
//with predictors A
void AddRegression(int nMaxRegs, LPRegression regressions, int numRegressor, int regressor, double logmarglikLap,
                   double logmarglikMon, double beta0, double beta1)
{
    int i, j = 0;
    
    LPRegression p = regressions;
    LPRegression pnext = p->Next;
    
    while(NULL!=pnext && j<nMaxRegs)
    {
        //return if we have previously found this regression
        if(IsSameRegression(numRegressor, regressor, pnext->numRegressor, pnext->regressor))
        {
            return;
        }
        
        //go to the next element in the list if the current
        //regression has a larger log marginal likelihood than
        //the new regression A
        if(pnext->logmarglikMon > logmarglikMon)
        {
            p = pnext;
            pnext = p->Next;
        }
        else //otherwise stop; this is where we insert the new regression
        {
            break;
        }
        j++;
    }
    
    if(j == nMaxRegs)
    {
        return;
    }
    
    //create a new element of the list
    LPRegression regressionNew = new Regression;
    regressionNew->numRegressor = numRegressor;
    regressionNew->regressor = regressor;
    regressionNew->logmarglikMon = logmarglikMon;
    regressionNew->logmarglikLap = logmarglikLap;
    regressionNew->beta0 = beta0;
    regressionNew->beta1 = beta1;
    
    //insert the new element in the list
    p->Next = regressionNew;
    regressionNew->Next = pnext;
    
    //arrive the last regression
    while(NULL!=pnext && j < nMaxRegs)
    {
        p = pnext;
        pnext = p->Next;
        j++;
    }
    
    //delete the worst regression.
    if(j == nMaxRegs)
    {
        DeleteLastRegression(regressions);
    }
    
    return;
}
