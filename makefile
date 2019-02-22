MPI_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas
MPI_INCLUDE = -I/usr/include/
MPI_LIB = -L/usr/lib/
MPI_OBJS = FinalProject

all:	${MPI_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	gcc -g -c matrices.cpp -o matrices.o ${MPI_INCLUDE} 

regmodels.o: regmodels.cpp regmodels.h
	gcc -g -c regmodels.cpp -o regmodels.o ${MPI_INCLUDE} 

main.o: main.cpp matrices.h
	mpic++ -g -c main.cpp -o main.o ${MPI_INCLUDE} ${MPI_LIB}

FinalProject: main.o matrices.o regmodels.o
	mpic++ main.o regmodels.o matrices.o -o FinalProject ${MPI_LIB} ${MPI_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${MPI_OBJS}