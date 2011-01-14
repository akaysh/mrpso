#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "tests.h"
#include "helper.h"

__global__ void FindGBests(int numSwarms, int numParticles, int numTasks, float *pBest, float *pBestPosition, float *gBest, float *gBestPosition)
{
	extern __shared__ float gBestCandidates[];
	__shared__ float *gBestCandidateIndices;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int gBestIndex;
	unsigned int i;
	int swarmOffset;
	int particleOffset;

	gBestCandidateIndices = &gBestCandidates[blockDim.x];

	//Have each thread load values into the gBestCandidates
	if (threadIdx.x < numParticles)
	{
		gBestCandidates[threadIdx.x] = pBest[__mul24(blockIdx.x, numParticles) + threadIdx.x];
		gBestCandidateIndices[threadIdx.x] = threadIdx.x;
	}

	__syncthreads();

	//Now perform a quick parallel reduction to find the minimum GBest value
	for (i = blockDim.x / 2; i > 0; i >>= 1)
	{
		if (threadIdx.x < i)
		{
			if (gBestCandidates[threadIdx.x] > gBestCandidates[threadIdx.x + i]) //Then replace the value!
			{
				gBestCandidateIndices[threadIdx.x] = gBestCandidateIndices[threadIdx.x + i];
				gBestCandidates[threadIdx.x] = gBestCandidates[threadIdx.x + i];
			}
		}

		__syncthreads();
	}

	gBestIndex = gBestCandidateIndices[0];
	swarmOffset = blockIdx.x * numParticles * numTasks;	
	particleOffset = gBestIndex * numTasks;	

	if (threadIdx.x == 0)
	{
		if (gBest[blockIdx.x] > gBestCandidates[0])
			gBest[blockIdx.x] = gBestCandidates[0];

		for (i = 0; i < numTasks; i++)
			gBestPosition[blockIdx.x * numTasks + i] = pBestPosition[swarmOffset + particleOffset + i];
	}	
}

void FindGlobalBest(float *pBest, float *pBestPositionVector, int numParticles, float *currGBest, float *gBestPositionVector, int numTasks)
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

int GlobalBestDeterminationTest()
{
	int passed = 1;
	int i, j, k;
	float *hPBest, *dPBest, *cpuPBest;
	float *hGBest, *dGBest, *cpuGBest;
	float *hPBestPosition, *dPBestPosition, *cpuPBestPosition;
	float *hGBestPosition, *dGBestPosition, *cpuGBestPosition;
	int numSwarms, numParticles, numTasks, numMachines;
	int numBlocks, threadsPerBlock;

	numSwarms = 40;
	numParticles = 64;
	numTasks = 200;
	numMachines = 8;
	threadsPerBlock = numParticles;
	numBlocks = numSwarms;

	printf("\tRunning GPU global best test...\n");

	srand((unsigned int) time(NULL));

	hPBest = (float *) malloc(numSwarms * numParticles * sizeof(float));
	cpuPBest = (float *) malloc(numSwarms * numParticles * sizeof(float));
	hGBest = (float *) malloc(numSwarms * sizeof(float));
	cpuGBest = (float *) malloc(numSwarms *  sizeof(float));

	hPBestPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	cpuPBestPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	hGBestPosition = (float *) malloc(numSwarms * numTasks * sizeof(float));
	cpuGBestPosition = (float *) malloc(numSwarms * numTasks * sizeof(float));

	cudaMalloc((void **) &dPBest, numSwarms * numParticles * sizeof(float));
	cudaMalloc((void **) &dGBest, numSwarms * sizeof(float));
	cudaMalloc((void **) &dPBestPosition, numSwarms * numParticles * numTasks * sizeof(float));
	cudaMalloc((void **) &dGBestPosition, numSwarms * numTasks * sizeof(float));

	//Randomly generate our PBest values and positions
	for (i = 0; i < numSwarms * numParticles; i++)
	{
		hPBest[i] = rand() % 1000 + 1;
		cpuPBest[i] = hPBest[i];

		for (j = 0; j < numTasks; j++)
		{
			hPBestPosition[i * numTasks + j] = rand() % numMachines;
			cpuPBestPosition[i * numTasks + j] = hPBestPosition[i * numTasks + j];
		}
	}

	//Randomly generate our GBest values and positions.
	for (i = 0; i < numSwarms; i++)
	{
		hGBest[i] = rand() % 1000 + 1;

		cpuGBest[i] = hGBest[i];

		for (j = 0; j < numTasks; j++)
		{
			hGBestPosition[i * numTasks + j] = rand() % numMachines;
			cpuGBestPosition[i * numTasks + j] = hGBestPosition[i * numTasks + j];
		}
	}	

	//Dump the data to the GPU
	cudaMemcpy(dPBest, hPBest, numSwarms * numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGBest, hGBest, numSwarms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dPBestPosition, hPBestPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGBestPosition, hGBestPosition, numSwarms * numTasks * sizeof(float), cudaMemcpyHostToDevice);

	//Compute the reference solution
	for (i = 0; i < numSwarms; i++)
	{
		FindGlobalBest(&cpuPBest[i * numParticles], &cpuPBestPosition[i * numParticles * numTasks], numParticles, &cpuGBest[i], &cpuGBestPosition[i * numTasks], numTasks);
	}

	//Compute the GPU solution
	FindGBests<<<numBlocks, threadsPerBlock, threadsPerBlock * 2 * sizeof(float)>>>(numSwarms, numParticles, numTasks, dPBest, dPBestPosition, 
																				    dGBest, dGBestPosition);
	cudaThreadSynchronize();

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

		for (j = 0; j < numParticles; j++)
		{
			if (abs(hPBest[i * numParticles + j] - cpuPBest[i * numParticles + j]) > ACCEPTED_DELTA)
			{
				printf("\t[ERROR] - GPU particle best value for particle [%d:%d] was: %f (expected: %f)\n", j, i, 
					   hPBest[i * numParticles + j], cpuPBest[i * numParticles + j]);
				passed = 0;
			}

		}

	}

	PrintTestResults(passed);

	return passed;
}

void RunGlobalBestTests()
{
	int passed = 1;

	printf("\nStarting GPU global best tests...\n\n");

	passed &= GlobalBestDeterminationTest();

	if (passed)
		printf("[PASSED] All global best tests passed!\n\n");
	else
		printf("[FAILED] Global best tests failed!\n\n");
}