#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "tests.h"

void TestSwarmInitialization()
{
	int i;
	int numTasks, numMachines;
	RunConfiguration *run;
	float *hPosition, *dPosition, *cpuPosition;
	float *hVelocity, *dVelocity, *cpuVelocity;
	float *hRand, *dRand;

	run = (RunConfiguration *) malloc(sizeof(RunConfiguration));
	
	run->numSwarms = 30;
	run->numParticles = 64;
	run->numIterations = 1000;
	numTasks = 1000;
	numMachines = 100;
	
	hPosition = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	hVelocity = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	cpuPosition = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	cpuVelocity = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	hRand = (float *) malloc(run->numSwarms * run->numParticles * numTasks * 2);





}

void RunSwarmInitializationTests()
{
	int passed = 1;

	printf("\nStarting GPU swarm initialization tests...\n\n");

	if (passed)
		printf("[PASSED] All swarm initialization tests passed!\n\n");
	else
		printf("[FAILED] Swarm initialization tests failed!\n\n");
}