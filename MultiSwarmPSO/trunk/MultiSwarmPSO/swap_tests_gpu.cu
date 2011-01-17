#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "tests.h"
#include "helper.h"
#include "gpu_pso.h"

int floatcomp(const void* elem1, const void* elem2)
{
    if(*(const float*)elem1 < *(const float*)elem2)
        return -1;
    return *(const float*)elem1 > *(const float*)elem2;
}

int TestSwapParticles()
{
	int i, j, k, mySwarmOffset, previousSwarmOffset, neighborSwarmOffset, previousSwarmValue, neighborSwarmValue;
	int passed = 1;
	Particle *particles;
	float *hPosition, *dPosition, *hVelocity, *dVelocity;
	int *bestListing;
	int *worstListing;
	int *dBestSwapIndices, *dWorstSwapIndices;
	float *fitnesses;
	int numParticles;
	int numToSwap;
	int numSwarms;
	int numTasks;
	int numMachines;
	float currFitness;
	int index;
	int threadsPerBlock, numBlocks;	

	numParticles = 5;
	numToSwap = 2;
	numSwarms = 2;
	numTasks = 10;
	numMachines = 4;

	printf("\tRunning particle swap test...\n");

	hPosition = (float *) malloc(numParticles * numSwarms * numTasks * sizeof(float));
	hVelocity = (float *) malloc(numParticles * numSwarms * numTasks * sizeof(float));
	bestListing = (int *) malloc(numToSwap * numSwarms * sizeof(int));
	worstListing = (int *) malloc(numToSwap * numSwarms * sizeof(int));
	fitnesses = (float *) malloc(numParticles * numSwarms * sizeof(float));

	cudaMalloc((void **) &dPosition, numParticles * numSwarms * numTasks * sizeof(float));
	cudaMalloc((void **) &dVelocity, numParticles * numSwarms * numTasks * sizeof(float));
	cudaMalloc((void **) &dBestSwapIndices, numToSwap * numSwarms * sizeof(int));
	cudaMalloc((void **) &dWorstSwapIndices, numToSwap * numSwarms * sizeof(int));

	srand((unsigned int) time(NULL));

	//Initialize our Particles
	particles = (Particle *) malloc(numParticles * numSwarms * sizeof(Particle));
	
	for (i = 0; i < numParticles * numSwarms; i++)
	{
		fitnesses[i] = (rand() % 1000) + 1;
		particles[i].fitness = fitnesses[i];
	}

	//Locate the top numToSwap and worst numToSwap Particles in each swarm by qsorting
	//the fitnesses for each swarm and dumping them into the relevant best/worst listing.
	for (i = 0; i < numSwarms; i++)
	{
		qsort(&fitnesses[numParticles * i], numParticles, sizeof(float), floatcomp);

		index = i * numToSwap;

		for (j = 0; j < numToSwap; j++)
		{
			currFitness = fitnesses[i * numParticles + j];			

			//Search for this fitness value in the particles to get the 'real' index.
			for (k = 0; k < numParticles; k++)
			{
				if (abs(particles[i * numParticles + k].fitness - currFitness) < ACCEPTED_DELTA) //Then we found it, mark the index.
				{
					bestListing[index++] = k;
					break;
				}
			}
		}

		index = i * numToSwap;

		for (j = numParticles - numToSwap; j < numParticles; j++)
		{
			currFitness = fitnesses[i * numParticles + j];			

			//Search for this fitness value in the particles to get the 'real' index.
			for (k = 0; k < numParticles; k++)
			{
				if (abs(particles[i * numParticles + k].fitness - currFitness) < ACCEPTED_DELTA) //Then we found it, mark the index.
				{
					worstListing[index++] = k;
					break;
				}
			}
		}
	}

	//Generate some simple values that we can track for the position and velocities...
	for (i = 0; i < numSwarms; i++)
	{
		for (j = 0; j < numParticles; j++)
		{
			for (k = 0; k < numTasks; k++)
			{
				hPosition[(i * numParticles * numTasks) + j * numTasks + k] = (float) i;
				hVelocity[(i * numParticles * numTasks) + j * numTasks + k] = (float) i;
			}
		}
	}

	//Copy the memory over to the GPU
	cudaMemcpy(dPosition, hPosition, numParticles * numSwarms * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocity, hVelocity, numParticles * numSwarms * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dBestSwapIndices, bestListing, numToSwap * numSwarms * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dWorstSwapIndices, worstListing, numToSwap * numSwarms * sizeof(int), cudaMemcpyHostToDevice);

	threadsPerBlock = 32;
	numBlocks = CalcNumBlocks(numSwarms * numToSwap, threadsPerBlock);

	SwapBestParticles<<<numBlocks, threadsPerBlock>>>(numSwarms, numParticles, numTasks, numToSwap, dBestSwapIndices, dWorstSwapIndices, dPosition, dVelocity);
	cudaThreadSynchronize();

	//Copy the data back
	cudaMemcpy(hPosition, dPosition, numParticles * numSwarms * numTasks * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hVelocity, dVelocity, numParticles * numSwarms * numTasks * sizeof(float), cudaMemcpyDeviceToHost);

	//Ensure that the correct modified numbers were added.
	for (i = 0; i < numSwarms; i++)
	{
		mySwarmOffset = i * numParticles * numTasks;

		previousSwarmValue = i != 0 ? i - 1 : numSwarms - 1;
		previousSwarmOffset = previousSwarmValue * numParticles * numTasks;

		neighborSwarmValue = i < numSwarms - 1 ? i + 1 : 0;
		neighborSwarmOffset = neighborSwarmValue * numParticles * numTasks;

		//For this swarm ensure our 'best' particles position and velocity values are now equal to our neighboring swarm's values.
		//				 ensure our 'worst' particles position and velocity values are now equal to the "previous" swarm's values.
		for (j = 0; j < numToSwap; j++)
		{
			for (k = 0; k < numTasks; k++)
			{
				if(abs(hPosition[mySwarmOffset + (bestListing[j + (i * numToSwap)] * numTasks) + k] - neighborSwarmValue) > ACCEPTED_DELTA)
				{
					printf("\t[ERROR] - GPU Position value for swarm %d, particle %d, element %d was: %f (expected: %d)\n", i, bestListing[j], k,
						                          hPosition[mySwarmOffset + (bestListing[j * (i * numToSwap)] * numTasks) + k], neighborSwarmValue);
					passed = 0;
				}

				if(abs(hVelocity[mySwarmOffset + (bestListing[j + (i * numToSwap)] * numTasks) + k] - neighborSwarmValue) > ACCEPTED_DELTA)
				{
					printf("\t[ERROR] - GPU Velocity value for swarm %d, particle %d, element %d was: %f (expected: %d)\n", i, bestListing[j], k,
						                          hVelocity[mySwarmOffset + (bestListing[j * (i * numToSwap)] * numTasks) + k], neighborSwarmValue);
					passed = 0;
				}
			}
		}
	}

	PrintTestResults(passed);

	return passed;
}

void RunSwapTests()
{
	int passed = 1;

	printf("\nStarting GPU swap tests...\n\n");

	passed &= TestSwapParticles();

	if (passed)
		printf("[PASSED] All swap tests passed!\n\n");
	else
		printf("[FAILED] Swap tests failed!\n\n");
}