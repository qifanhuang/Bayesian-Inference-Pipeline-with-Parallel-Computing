/*
 Qifan Huang's final project.
 
 Run the program using the command:
 
 mpirun -np 6 FinalProject
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "matrices.h"
#include "regmodels.h"

//For MPI communication
#define GETPI 1
#define DIETAG 0


//Used to determine MASTER or SLAVE
static int myrank;

int nvariables = 61;
int nobservations = 148;

//Function Declarations
void master();
void slave(int slavename, gsl_matrix* data);


int main(int argc,char** argv)
{
    
    ///////////////////////////
    // START THE MPI SESSION //
    ///////////////////////////
    MPI_Init(&argc, &argv);
    
    /////////////////////////////////////
    // What is the ID for the process? //
    /////////////////////////////////////
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //load data
    gsl_matrix* data = gsl_matrix_alloc(nobservations, nvariables);
    char datafilename[] = "534finalprojectdata.txt";
    FILE* datafile = fopen(datafilename, "r");
    
    if(NULL == datafile)
    {
        fprintf(stderr, "Cannot open data file [%s]\n", datafilename);
    }
    if (0 != gsl_matrix_fscanf(datafile, data))
    {
        fprintf(stderr, "File [%s] does not have the required format.\n", datafilename);
    }
    fclose(datafile);
    
    
    if(myrank==0)
    {
        master();
    }
    else
    {
        slave(myrank, data);
    }
    
    gsl_matrix_free(data);
    
    //Finalize the MPI session
    MPI_Finalize();
    
    return(1);
}






void master()
{
    int var;                                   //to loop over the variables
    int rank;                                  //another looping variable
    int ntasks;                                //the total number of slaves
    int jobsRunning;                           //how many slaves we have working
    int work[1];                               //information to send to the slaves
    double workresults[5];                     //info received from the slaves
    char outputfilename[] = "BestFiveRegressions.txt"; //name of output file
    MPI_Status status;	                       //MPI information
    
    //create an empty single list
    LPRegression regressions = new Regression;
    regressions->Next = NULL;
    
    //Find out how many slaves there are
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
    fprintf(stderr, "Total Number of processors = %d\n",ntasks);
    
    jobsRunning = 1;
    
    for(var=0; var<nvariables-1; var++)
    {
        // This will tell the slave which variable(regressor) to work on
        work[0] = var;
        
        if(jobsRunning < ntasks)               //Do we have an available processor?
        {
            // Send out a work request
            MPI_Send(&work, 	               //the vector with the variable
                     1, 	                   //the size of the vector
                     MPI_INT,	               //the type of the vector
                     jobsRunning,	           //the ID of the slave to use
                     GETPI,	                   //tells the slave what to do
                     MPI_COMM_WORLD);          //send the request out to anyone who is available
            
            printf("Master sends out work request [%d] to slave [%d]\n",
                   work[0],jobsRunning);
            
            // Increase the # of processors in use
            jobsRunning++;
        }
        else                                   //all the processors are in use!
        {
            MPI_Recv(workresults,	           //where to store the results
                     5,		                   //the size of the vector
                     MPI_DOUBLE,	           //the type of the vector
                     MPI_ANY_SOURCE,
                     MPI_ANY_TAG,
                     MPI_COMM_WORLD,
                     &status);                 //lets us know which processor
            printf("Master has received the result of work request [%d] from slave [%d]\n",
                   (int)workresults[0],status.MPI_SOURCE);
            
            //save the results received
            AddRegression(5, regressions, 1, (int)workresults[0]+1, workresults[1], workresults[2], workresults[3], workresults[4]);
            
            printf("Master sends out work request [%d] to slave [%d]\n",
                   work[0],status.MPI_SOURCE);
            // Send out a new work order to the processors that just
            // returned
            MPI_Send(&work,
                     1,
                     MPI_INT,
                     status.MPI_SOURCE,        //the slave that just returned
                     GETPI,
                     MPI_COMM_WORLD);
        }                                      //using all the processors
    }                                          //loop over all the variables
    
    ///////////////////////////////////////////////////////////////
    // NOTE: we still have some work requests out that need to be
    // collected. Collect those results now!
    ///////////////////////////////////////////////////////////////
    
    // loop over all the slaves
    for(rank=1; rank<jobsRunning; rank++)
    {
        MPI_Recv(workresults,
                 5,
                 MPI_DOUBLE,
                 MPI_ANY_SOURCE,               //whoever is ready to report back
                 MPI_ANY_TAG,
                 MPI_COMM_WORLD,
                 &status);
        
        printf("Master has received the result of work request [%d]\n",
               (int)workresults[0]);
        
        //save the results received
        AddRegression(5, regressions, 1, (int)workresults[0]+1, workresults[1], workresults[2], workresults[3], workresults[4]);
    }
    
    printf("Tell the slave to die\n");
    
    // Shut down the slave processes
    for(rank=1; rank<ntasks; rank++)
    {
        printf("Master is killing slave [%d]\n",rank);
        MPI_Send(0,
                 0,
                 MPI_INT,
                 rank,		                  //shutdown this particular node
                 DIETAG,		              //tell it to shutdown
                 MPI_COMM_WORLD);
    }
    
    
    printf("got to the end of Master code\n");
    
    // Save the list
    SaveRegressions(outputfilename, regressions);
    
    DeleteAllRegressions(regressions);
    delete regressions;
    regressions = NULL;
    
    // return to the main function
    return;
}





void slave(int slavename, gsl_matrix* data)
{
    int work[1];                             //the inputs from the master
    double workresults[5];                   //the outputs for the master
    MPI_Status status;                       //for MPI communication
    double logmarglikLap, logmarglikMon;
    gsl_vector* betaPosterior = gsl_vector_alloc(2);
    gsl_vector* betaMode = gsl_vector_alloc(2);
    
    // initialize the GSL generator
    const gsl_rng_type* T;
    gsl_rng* r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    
    // the slave listens for instructions...
    int notDone = 1;
    while(notDone)
    {
        printf("Slave %d is waiting\n",slavename);
        MPI_Recv(&work,		                //the inputs from the master
                 1,		                    //the size of the inputs
                 MPI_INT,		            //the type of the inputs
                 0,		                    //from the MASTER node (rank=0)
                 MPI_ANY_TAG,	            //any type of order is fine
                 MPI_COMM_WORLD,
                 &status);
        printf("Slave %d just received smth\n",slavename);
        
        // switch on the type of work request
        switch(status.MPI_TAG)
        {
            case GETPI:
                printf("Slave %d has received work request [%d]\n",
                       slavename,work[0]);
                
                betaMode = getcoefNR(60, work[0], data, 1e-6);
                logmarglikLap = getLaplace(60, work[0], data, betaMode);
                logmarglikMon = getMonteCarlo(10000, r, 60, work[0], data);
                betaPosterior = getcoefMH(10000, r, 60, work[0], data, betaMode);
                
                workresults[0] = (double)work[0];
                workresults[1] = logmarglikLap;
                workresults[2] = logmarglikMon;
                workresults[3] = gsl_vector_get(betaPosterior, 0);
                workresults[4] = gsl_vector_get(betaPosterior, 1);
                
                // Send the results
                MPI_Send(&workresults,
                         5,
                         MPI_DOUBLE,
                         0,		           //send it to the master
                         0,		           //doesn't need a TAG
                         MPI_COMM_WORLD);
                
                printf("Slave %d finished processing work request [%d]\n",
                       slavename,work[0]);
                break;
                
            case DIETAG:
                printf("Slave %d was told to die\n",slavename);
                return;
                
            default:
                notDone = 0;
                printf("The slave code should never get here.\n");
                return;
        }
    }
    
    //free memory
    gsl_vector_free(betaMode);
    gsl_vector_free(betaPosterior);
    gsl_rng_free(r);
    return;
}

