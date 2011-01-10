#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "tests.h"
#include "helper.h"
#include "gpu_pso.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

__device__ int GetDiscreteCoordT1(float val)
{
	return   floorf(val);
}

/* Unfortunately, we cannot do external calls to device code, so we have to copy this here under a DIFFERENT name(!!!)...
 * Thanks Nvidia!
 */
__device__ float CalcMakespanT(int numTasks, int numMachines, float *matching, float *scratch)
{
	int i;
	float makespan;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int taskOffset, machineOffset;
	float matchingVal;
	float val;
	
	makespan = 0.0f;
	taskOffset = __mul24(threadID, numTasks);
	machineOffset = __mul24(threadID, numMachines);

	//Clear our scratch table
	for (i = 0; i < numTasks; i++)
		scratch[machineOffset + GetDiscreteCoordT1(matching[taskOffset + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		matchingVal = matching[taskOffset + i];

		scratch[machineOffset + GetDiscreteCoordT1(matchingVal)] += tex2D(texETCMatrix, matchingVal, (float) i);
		val = scratch[machineOffset + GetDiscreteCoordT1(matchingVal)];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

__global__ void TestMakespan(int numTasks, int numMachines, int numMatchings, float *matching, float *scratch, float *outVal)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < numMatchings)
		outVal[threadID] = CalcMakespanT(numTasks, numMachines, matching, scratch);
}

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
					printf("[ERROR] - GPU Position value for swarm %d, particle %d, element %d was: %f (expected: %d)\n", i, bestListing[j], k,
						                          hPosition[mySwarmOffset + (bestListing[j * (i * numToSwap)] * numTasks) + k], neighborSwarmValue);
					passed = 0;
				}

				if(abs(hVelocity[mySwarmOffset + (bestListing[j + (i * numToSwap)] * numTasks) + k] - neighborSwarmValue) > ACCEPTED_DELTA)
				{
					printf("[ERROR] - GPU Velocity value for swarm %d, particle %d, element %d was: %f (expected: %d)\n", i, bestListing[j], k,
						                          hVelocity[mySwarmOffset + (bestListing[j * (i * numToSwap)] * numTasks) + k], neighborSwarmValue);
					passed = 0;
				}
			}
		}
	}

	PrintTestResults(passed);

	return passed;
}

int TestGPUMakespan()
{
	int i;
	int passed = 1;
	cudaArray *cuArray;
	float *dOut, *matching, *scratch;
	float *hMatching, *hScratch;
	int numMatchings;
	int threadsPerBlock, numBlocks;
	float *cpuMakespans, *gpuMakespans;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	BuildMachineList("machines8.txt");
	BuildTaskList("tasks80.txt");
	GenerateETCMatrix();

	numMatchings = 128;
	threadsPerBlock = 64;
	numBlocks = CalcNumBlocks(numMatchings, threadsPerBlock);

	printf("Running GPU Makespan Test...\n");

	srand((unsigned int) time(NULL));

	hMatching = (float *) calloc(numMatchings * GetNumTasks(), sizeof(float));
	hScratch = (float *) calloc(numMatchings * GetNumMachines(), sizeof(float));
	cpuMakespans = (float *) malloc(numMatchings * sizeof(float));
	gpuMakespans = (float *) malloc(numMatchings * sizeof(float));

	for (i = 0; i < numMatchings * GetNumTasks(); i++)
		hMatching[i] = (float) (rand() % (GetNumMachines() * 100)) / 100.0f;

	//Compute the makespans on the CPU
	for (i = 0; i < numMatchings; i++)
		cpuMakespans[i] = ComputeMakespan(&hMatching[i * GetNumTasks()], GetNumTasks());

	cudaMalloc((void **)&dOut, sizeof(float) * numMatchings );
	cudaMalloc((void **)&matching, sizeof(float) * numMatchings * GetNumTasks() );
	cudaMalloc((void **)&scratch, sizeof(float) * numMatchings * GetNumMachines() );

	texETCMatrix.normalized = false;
	texETCMatrix.filterMode = cudaFilterModePoint;
	texETCMatrix.addressMode[0] = cudaAddressModeClamp;
    texETCMatrix.addressMode[1] = cudaAddressModeClamp;


	cudaMallocArray(&cuArray, &channelDesc, GetNumMachines(), GetNumTasks());
	cudaMemcpyToArray(cuArray, 0, 0, hETCMatrix, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texETCMatrix, cuArray, channelDesc);

	cudaMemcpy(matching, hMatching, sizeof(float) * numMatchings * GetNumTasks(), cudaMemcpyHostToDevice);
	cudaMemcpy(scratch, hScratch, sizeof(float) * numMatchings * GetNumMachines(), cudaMemcpyHostToDevice);

	TestMakespan<<<numBlocks, threadsPerBlock>>>(GetNumTasks(), GetNumMachines(), numMatchings, matching, scratch, dOut);

	cudaMemcpy(gpuMakespans, dOut, sizeof(float) * numMatchings , cudaMemcpyDeviceToHost);

	for (i = 0; i < numMatchings; i++)
	{
		if (abs(gpuMakespans[i] - cpuMakespans[i]) > ACCEPTED_DELTA)
		{
			printf("[ERROR] - %d GPU Makespan was: %f (expected: %f)\n", i, gpuMakespans[i], cpuMakespans[i]);
			passed = 0;
		}
	}

	PrintTestResults(passed);

	free(hMatching);
	free(hScratch);
	free(cpuMakespans);
	free(gpuMakespans);
	cudaFree(dOut);
	cudaFree(matching);
	cudaFree(scratch);
	cudaFreeArray(cuArray);

	return passed;
}

void RunSwarmFunctionTests()
{
	int passed = 1;

	printf("Starting GPU Swarm Function tests...\n\n");

	passed &= TestSwapParticles();
	passed &= TestGPUMakespan();

	if (passed)
		printf("All swarm function tests passed!\n\n");
	else
		printf("Swarm function tests failed!\n\n");


}