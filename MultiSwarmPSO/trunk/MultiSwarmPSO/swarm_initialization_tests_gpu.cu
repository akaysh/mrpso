#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "tests.h"
#include "gpu_pso.h"

int TestSwarmInitialization()
{
	int i;
	int passed = 1;
	int numTasks, numMachines;
	RunConfiguration *run;
	float *hPosition, *dPosition, *cpuPosition;
	float *hVelocity, *dVelocity, *cpuVelocity;
	float *hRand, *dRand;
	int numBlocks, threadsPerBlock;
	int randOffset;

	run = (RunConfiguration *) malloc(sizeof(RunConfiguration));

	printf("\tRunning GPU PSO initialization test...\n");
	
	run->numSwarms = 30;
	run->numParticles = 64;
	run->numIterations = 1000;
	numTasks = 1000;
	numMachines = 100;
	threadsPerBlock = 512;
	numBlocks = CalcNumBlocks(run->numSwarms * run->numParticles * run->numIterations, threadsPerBlock);
	
	hPosition = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	hVelocity = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	cpuPosition = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	cpuVelocity = (float *) malloc(run->numSwarms * run->numParticles * numTasks * sizeof(float));
	hRand = (float *) malloc(run->numSwarms * run->numParticles * numTasks * 2 * sizeof(float));

	cudaMalloc((void**) &dPosition, run->numSwarms * run->numParticles * numTasks * sizeof(float));
	cudaMalloc((void**) &dVelocity, run->numSwarms * run->numParticles * numTasks * sizeof(float));
	cudaMalloc((void**) &dRand, run->numSwarms * run->numParticles * numTasks * 2 * sizeof(float));

	//Generate our random numbers on the GPU and transfer them back to the CPU.
	GenerateRandsGPU(run->numSwarms * run->numParticles * numTasks * 2, dRand);
	cudaMemcpy(hRand, dRand, run->numSwarms * run->numParticles * numTasks * 2 * sizeof(float), cudaMemcpyDeviceToHost);

	randOffset = run->numSwarms * run->numParticles * numTasks;

	//Calculate the reference CPU solution
	for (i = 0; i < run->numSwarms * run->numParticles * numTasks; i++)
	{
		cpuPosition[i] = (numMachines - 1) * hRand[i];
		cpuVelocity[i] = (numMachines >> 1) * hRand[i + randOffset];
	}

	//Have the GPU perform the initialization...
	InitializeParticles<<<numBlocks, threadsPerBlock>>>(run->numSwarms * run->numParticles, numTasks, numMachines, dPosition, dVelocity, dRand);
	cudaThreadSynchronize();

	//Copy the GPU results back and compare them.
	cudaMemcpy(hPosition, dPosition, run->numSwarms * run->numParticles * numTasks * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hVelocity, dVelocity, run->numSwarms * run->numParticles * numTasks * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < run->numSwarms * run->numParticles * numTasks; i++)
	{
		if (abs(cpuPosition[i] - hPosition[i]) > ACCEPTED_DELTA)
		{
			printf("\t[ERROR] - %d GPU position was: %f (expected: %f)\n", i, hPosition[i], cpuPosition[i]);
			passed = 0;
		}

		if (abs(cpuVelocity[i] - hVelocity[i]) > ACCEPTED_DELTA)
		{
			printf("\t[ERROR] - %d GPU velocity was: %f (expected: %f)\n", i, hVelocity[i], cpuVelocity[i]);
			passed = 0;
		}
	}

	free(cpuPosition);
	free(cpuVelocity);
	free(hPosition);
	free(hVelocity);
	free(run);
	free(hRand);
	cudaFree(dRand);
	cudaFree(dPosition);
	cudaFree(dVelocity);

	PrintTestResults(passed);

	return passed;
}

void RunSwarmInitializationTests()
{
	int passed = 1;

	printf("\nStarting GPU swarm initialization tests...\n\n");

	passed &= TestSwarmInitialization();

	if (passed)
		printf("[PASSED] All swarm initialization tests passed!\n\n");
	else
		printf("[FAILED] Swarm initialization tests failed!\n\n");
}