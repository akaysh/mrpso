#ifndef _PSO_TESTS_H_
#define _PSO_TESTS_H_

#include <stdio.h>
#include <stdlib.h>

#define ACCEPTED_DELTA 0.00001

#define W 1.0f
#define WDECAY 0.995f
#define C1 2.0f
#define C2 1.4f

float *GetFCFSMapping(int numTasks, int numMachines);

int TestETCMatrixGeneration();
int TestFitnessAllDifferentMachines();
int TestFitnessSameMachines();

void TestMakespanResults(char *filename);

#endif
