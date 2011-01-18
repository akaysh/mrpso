#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tests.h"
#include "helper.h"
#include "gpu_pso.h"

void FindLocalBest(int numTasks, float *fitness, float *position, float *pBest, float *pBestPosition)
{
	int i;

	//Check if the local best needs updating...
	if (fitness[0] < pBest[0])
	{
		pBest[0] = fitness[0];

		//Copy the position!
		for (i = 0; i < numTasks; i++)
			pBestPosition[i] = position[i];
	}
}

void FindGlobalBests(float *pBest, float *pBestPositionVector, int numParticles, float *currGBest, float *gBestPositionVector, int numTasks)
{
	int i, particleIndex;

	particleIndex = -1;

	//Search for a new global best if one exists.
	for (i = 0; i < numParticles; i++)
	{
		if (pBest[i] < *currGBest)
		{
			*currGBest = pBest[i];
			particleIndex = i;
		}
	}

	//If we found a new global best, copy the position over from it.
	//We perform this step separate from the search as we don't want to
	//continually copy data if we find multiple global "bests" in the 
	//previous for loop.
	if (particleIndex >= 0)
	{
		for (i = 0; i < numTasks; i++)
			gBestPositionVector[i] = pBestPositionVector[(particleIndex * numTasks) + i];
	}
}

int TestLocalAndGlobalBestUpdate()
{
	int passed = 1;
	int i, j, k;
	float *hPosition, *dPosition;
	float *hPBest, *dPBest, *cpuPBest;
	float *hGBest, *dGBest, *cpuGBest;
	float *hPBestPosition, *dPBestPosition, *cpuPBestPosition;
	float *hGBestPosition, *dGBestPosition, *cpuGBestPosition;
	float *hFitness, *dFitness;
	int numSwarms, numParticles, numTasks, numMachines;
	int numBlocks, threadsPerBlock;

	numSwarms = 200;
	numParticles = 32;
	numTasks = 10;
	numMachines = 800;
	threadsPerBlock = numParticles;
	numBlocks = numSwarms;

	printf("\tRunning GPU best update test...\n");

	srand((unsigned int) time(NULL));

	hPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	hFitness = (float *) malloc(numSwarms * numParticles * sizeof(float));
	hPBest = (float *) malloc(numSwarms * numParticles * sizeof(float));
	cpuPBest = (float *) malloc(numSwarms * numParticles * sizeof(float));
	hGBest = (float *) malloc(numSwarms * sizeof(float));
	cpuGBest = (float *) malloc(numSwarms *  sizeof(float));

	hPBestPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	cpuPBestPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	hGBestPosition = (float *) malloc(numSwarms * numTasks * sizeof(float));
	cpuGBestPosition = (float *) malloc(numSwarms * numTasks * sizeof(float));

	cudaMalloc((void **) &dPosition, numSwarms * numParticles * numTasks * sizeof(float));
	cudaMalloc((void **) &dFitness, numSwarms * numParticles * sizeof(float));
	cudaMalloc((void **) &dPBest, numSwarms * numParticles * sizeof(float));
	cudaMalloc((void **) &dGBest, numSwarms * sizeof(float));
	cudaMalloc((void **) &dPBestPosition, numSwarms * numParticles * numTasks * sizeof(float));
	cudaMalloc((void **) &dGBestPosition, numSwarms * numTasks * sizeof(float));

	//Randomly generate our fitness and PBest values and positions
	for (i = 0; i < numSwarms * numParticles; i++)
	{
		hPBest[i] = rand() % 1000 + 100000;
		hFitness[i] = rand() % 1000000 + 1;
		cpuPBest[i] = hPBest[i];

		for (j = 0; j < numTasks; j++)
		{
			hPBestPosition[i * numTasks + j] = rand() % numMachines;
			cpuPBestPosition[i * numTasks + j] = hPBestPosition[i * numTasks + j];

			hPosition[i * numTasks + j] = rand() % numMachines;
		}
	}

	//Randomly generate our GBest values and positions.
	for (i = 0; i < numSwarms; i++)
	{
		hGBest[i] = rand() % 1000000 + rand() % 10000 + rand() % 100330 + 1;

		cpuGBest[i] = hGBest[i];

		for (j = 0; j < numTasks; j++)
		{
			hGBestPosition[i * numTasks + j] = rand() % numMachines;
			cpuGBestPosition[i * numTasks + j] = hGBestPosition[i * numTasks + j];
		}
	}	

	//Dump the data to the GPU
	cudaMemcpy(dPosition, hPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dFitness, hFitness, numSwarms * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dPBest, hPBest, numSwarms * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGBest, hGBest, numSwarms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dPBestPosition, hPBestPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGBestPosition, hGBestPosition, numSwarms * numTasks * sizeof(float), cudaMemcpyHostToDevice);

	//Compute the reference CPU solution
	for (i = 0; i < numSwarms; i++)
	{
		for (j = 0; j < numParticles; j++)
		{
			FindLocalBest(numTasks, &hFitness[i * numParticles + j], &hPosition[i * numParticles * numTasks + j * numTasks], 
				               &cpuPBest[i * numParticles + j], &cpuPBestPosition[i * numParticles * numTasks + j * numTasks]);
		}
		FindGlobalBests(&cpuPBest[i * numParticles], &cpuPBestPosition[i * numParticles * numTasks], numParticles, &cpuGBest[i], &cpuGBestPosition[i * numTasks], numTasks);
	}

	//Compute the GPU solution.
	UpdateBests<<<numBlocks, threadsPerBlock, numParticles * 2 * sizeof(float)>>>(numSwarms, numParticles, numTasks, dPBest, dPBestPosition, dGBest, dGBestPosition, dPosition, dFitness);
	cudaThreadSynchronize();

	//Dump the P and G best data back
	cudaMemcpy(hPBest, dPBest, numSwarms * numParticles * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hGBest, dGBest, numSwarms * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hPBestPosition, dPBestPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hGBestPosition, dGBestPosition, numSwarms * numTasks * sizeof(float), cudaMemcpyDeviceToHost);

	//Confirm the GBest and PBest results
	for (i = 0; i < numSwarms; i++)
	{
		if (abs(hGBest[i] - cpuGBest[i]) > ACCEPTED_DELTA)
		{
			printf("\t[ERROR] - GPU global best value for swarm %d was: %f (expected: %f)\n", i, hGBest[i], cpuGBest[i]);
			passed = 0;
		}

		for (j = 0; j < numTasks; j++)
		{
			if (abs(hGBestPosition[i * numTasks + j] - cpuGBestPosition[i * numTasks + j]) > ACCEPTED_DELTA)
			{
				printf("\t[ERROR] - GPU global best position for swarm %d, pos %d, was: %f (expected: %f)\n", i, j, 
					   hGBestPosition[i * numTasks + j], cpuGBestPosition[i * numTasks + j]);
				passed = 0;
			}
		}

		for (j = 0; j < numParticles; j++)
		{
			if (abs(hPBest[i * numParticles + j] - cpuPBest[i * numParticles + j]) > ACCEPTED_DELTA)
			{
				printf("\t[ERROR] - GPU particle best value for particle [%d:%d] was: %f (expected: %f)\n", j, i, 
					   hPBest[i * numParticles + j], cpuPBest[i * numParticles + j]);
				passed = 0;
			}

			for (k = 0; k < numTasks; k++)
			{
				if (abs(hPBestPosition[i * numParticles * numTasks + j * numTasks + k] - cpuPBestPosition[i * numParticles * numTasks + j * numTasks + k]) > ACCEPTED_DELTA)
				{
					printf("\t[ERROR] - GPU particle best position for swarm %d, particle %d, pos %d, was: %f (expected: %f)\n", i, j, k,
						   hPBestPosition[i * numParticles * numTasks + j * numTasks + k], cpuPBestPosition[i * numParticles * numTasks + j * numTasks + k]);
					passed = 0;
				}
			}
		}
	}

	PrintTestResults(passed);

	return passed;
}

void RunBestUpdateTests()
{
	int passed = 1;

	printf("\nStarting GPU best update tests...\n\n");

	passed &= TestLocalAndGlobalBestUpdate();

	if (passed)
		printf("[PASSED] All best update tests passed!\n\n");
	else
		printf("[FAILED] Best update tests failed!\n\n");
}