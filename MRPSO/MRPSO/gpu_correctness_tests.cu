#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include "tests.h"
#include "helper.h"
#include "gpu_pso.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

__constant__ float constETC;

void PrintTestResults(int passed)
{
	if (passed)
		printf("[SUCCESS] Test passed!\n\n");
	else
		printf("[FAILURE] Test failed!\n\n");
}

__device__ int GetDiscreteCoordT(float val)
{
	return   floorf(val);
}

__global__ void TestDiscrete(float *floatVals, int *outVals, int numVals)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < numVals)
		outVals[threadID] = GetDiscreteCoordT(floatVals[threadID]);
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
		scratch[machineOffset + GetDiscreteCoordT(matching[taskOffset + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		matchingVal = matching[taskOffset + i];

		scratch[machineOffset + GetDiscreteCoordT(matchingVal)] += tex2D(texETCMatrix, matchingVal, (float) i);
		val = scratch[machineOffset + GetDiscreteCoordT(matchingVal)];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

__global__ void TestTexture(int numTasks, int numMachines, float *outVals)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < numTasks * numMachines)
		outVals[idx] = tex2D(texETCMatrix, threadIdx.x, blockIdx.x);
}

__global__ void TestRandTexture(float *dVals, float *dOut, int numTasks, int numMachines)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID < numTasks)
		dOut[threadID] = tex2D(texETCMatrix, dVals[threadID], (float) (threadID));
}

__global__ void TestMakespan(int numTasks, int numMachines, int numMatchings, float *matching, float *scratch, float *outVal)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < numMatchings)
		outVal[threadID] = CalcMakespanT(numTasks, numMachines, matching, scratch);
}

int TestTextureReads()
{
	int i;
	int passed = 1;
	cudaArray *cuArray;
	float *dOut;
	float *gpuETCMatrix;

	printf("Running Texture Read Test...\n");
	
	BuildMachineList("machines8.txt");
	BuildTaskList("tasks80.txt");
	GenerateETCMatrix();

	gpuETCMatrix = (float *) malloc(GetNumMachines() * GetNumTasks() * sizeof(float));
	cudaMalloc((void **)&dOut, GetNumMachines() * GetNumTasks() * sizeof(float));

	cudaMallocArray(&cuArray, &texETCMatrix.channelDesc, GetNumMachines(), GetNumTasks());
	cudaMemcpyToArray(cuArray, 0, 0, hETCMatrix, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texETCMatrix, cuArray);

	texETCMatrix.normalized = false;
	texETCMatrix.filterMode = cudaFilterModePoint;

	TestTexture<<<80, 8>>>(GetNumTasks(), GetNumMachines(), dOut);

	cudaMemcpy(gpuETCMatrix, dOut, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyDeviceToHost);

	for (i = 0; i < GetNumTasks() * GetNumMachines(); i++)
	{
		if (gpuETCMatrix[i] - hETCMatrix[i] > ACCEPTED_DELTA)
		{
			printf("[ERROR] - GPU ETC Matrix was: %f (expected: %f)\n", gpuETCMatrix[i], hETCMatrix[i]);
			passed = 0;
		}
	}

	PrintTestResults(passed);

	free(gpuETCMatrix);
	FreeCPUMemory();
	cudaFree(dOut);
	cudaFreeArray(cuArray);	
	
	return passed;
}

int TestTextureReadsRandom()
{
	int i;
	int passed = 1;
	cudaArray *cuArray;
	float *dOut;
	float *gpuETCMatrix;
	float *hMatching, *dMatching;
	float *cpuOut, *gpuOut;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	int threadsPerBlock, numBlocks;

	threadsPerBlock = 64;

	printf("Running Texture Read Random Test...\n");
	
	BuildMachineList("machines8.txt");
	BuildTaskList("tasks80.txt");
	GenerateETCMatrix();

	srand((unsigned int) time(NULL));


	gpuETCMatrix = (float *) malloc(GetNumMachines() * GetNumTasks() * sizeof(float));
	hMatching = (float *) malloc(GetNumTasks() * sizeof(float));
	cpuOut = (float *) malloc(GetNumTasks() * sizeof(float));
	gpuOut = (float *) malloc(GetNumTasks() * sizeof(float));

	cudaMalloc((void **)&dOut, GetNumMachines() * GetNumTasks() * sizeof(float));
	cudaMalloc((void **)&dMatching, GetNumTasks() * sizeof(float));
	cudaMalloc((void **)&dOut, GetNumTasks() * sizeof(float));

	texETCMatrix.normalized = false;
	texETCMatrix.filterMode = cudaFilterModePoint;
	texETCMatrix.addressMode[0] = cudaAddressModeClamp;
    texETCMatrix.addressMode[1] = cudaAddressModeClamp;

	cudaMallocArray(&cuArray, &channelDesc, GetNumMachines(), GetNumTasks());
	cudaMemcpyToArray(cuArray, 0, 0, hETCMatrix, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texETCMatrix, cuArray, channelDesc);

	for (i = 0; i < GetNumTasks(); i++)
		hMatching[i] = (float) (rand() % ((GetNumMachines() - 1) * 100)) / 100.0f;

	cudaMemcpy(dMatching, hMatching, GetNumTasks() * sizeof(float), cudaMemcpyHostToDevice);

	numBlocks = CalcNumBlocks(GetNumTasks(), threadsPerBlock);

	TestRandTexture<<<numBlocks, threadsPerBlock>>>(dMatching, dOut, GetNumTasks(), GetNumMachines());

	cudaMemcpy(gpuOut, dOut, GetNumTasks() * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < GetNumTasks(); i++)
		cpuOut[i] = hETCMatrix[(i * GetNumMachines()) + DiscreteCoord(hMatching[i])];

	for (i = 0; i < GetNumTasks(); i++)
	{
		if (abs(gpuOut[i] - cpuOut[i]) > ACCEPTED_DELTA)
		{
			printf("[ERROR] - %d GPU ETC Matrix was: %f (expected: %f)\n", i, gpuOut[i], cpuOut[i]);
			printf("\tOriginal matching value used: %f\n", hMatching[i]);
			passed = 0;
		}
	}

	PrintTestResults(passed);

	free(gpuETCMatrix);
	FreeCPUMemory();
	cudaFree(dOut);
	cudaFreeArray(cuArray);	
	
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

int TestDiscreteCoord()
{
	int i;
	int passed = 1;
	float *hFloats, *dFloats;
	int *dOut;
	int *hInts, *dInts;
	int numVals = 1024;
	int threadsPerBlock, numBlocks;

	threadsPerBlock = 512;

	srand((unsigned int) time(NULL));

	printf("Running GPU Discrete Coordinate Test...\n");

	hFloats = (float *) malloc(numVals * sizeof(float));
	hInts = (int *) malloc(numVals * sizeof(int));
	dInts = (int *) malloc(numVals * sizeof(int));

	cudaMalloc((void **) &dFloats, numVals * sizeof(float));
	cudaMalloc((void **) &dOut, numVals * sizeof(int));

	numBlocks = CalcNumBlocks(numVals, threadsPerBlock);

	for (i = 0; i < numVals; i++)
	{
		hFloats[i] = (float) (rand() % (100 * 100)) / 100.0f;
		hInts[i] = DiscreteCoord(hFloats[i]);
	}

	cudaMemcpy(dFloats, hFloats, sizeof(float) * numVals , cudaMemcpyHostToDevice);

	TestDiscrete<<<numBlocks, threadsPerBlock>>>(dFloats, dOut, numVals);

	cudaMemcpy(dInts, dOut, sizeof(int) * numVals , cudaMemcpyDeviceToHost);

	for (i = 0; i < numVals; i++)
	{
		if (hInts[i] != dInts[i])
		{
			printf("[ERROR] - GPU Int was: %d (expected: %d)\n", dInts[i], hInts[i]);
			passed = 0;
		}
	}

	PrintTestResults(passed);

	free(hFloats);
	free(hInts);
	free(dInts);
	cudaFree(dFloats);
	cudaFree(dOut);
	
	return passed;
}

int floatcomp(const void* elem1, const void* elem2)
{
    if(*(const float*)elem1 < *(const float*)elem2)
        return -1;
    return *(const float*)elem1 > *(const float*)elem2;
}


int TestSwapParticles()
{
	int i, j, k;
	int passed = 1;
	Particle *particles;
	float *hPosition, *dPosition, *hVelocity, *dVelocity;
	int *bestListing;
	int *worstListing;
	int *hBestSwapIndices, *hWorstSwapIndices, *dBestSwapIndices, *dWorstSwapIndices;
	float *fitnesses;
	int numParticles = 5;
	int numToSwap = 2;
	int numSwarms = 2;
	int numTasks = 10;
	int numMachines = 4;
	float currFitness;
	int index;

	srand((unsigned int) time(NULL));

	hPosition = (float *) calloc(numParticles * numSwarms, sizeof(float));
	hVelocity = (float *) calloc(numParticles * numSwarms, sizeof(float));
	bestListing = (int *) calloc(numToSwap * numSwarms, sizeof(int));
	worstListing = (int *) calloc(numToSwap * numSwarms, sizeof(int));
	hBestSwapIndices = (int *) calloc(numToSwap * numSwarms, sizeof(int));
	hWorstSwapIndices = (int *) calloc(numToSwap * numSwarms, sizeof(int));
	fitnesses = (float *) malloc(numParticles * numSwarms * sizeof(float));

	cudaMalloc((void **) &dPosition, numParticles * numSwarms * sizeof(float));
	cudaMalloc((void **) &dVelocity, numParticles * numSwarms * sizeof(float));
	cudaMalloc((void **) &dBestSwapIndices, numToSwap * numSwarms * sizeof(int));
	cudaMalloc((void **) &dWorstSwapIndices, numToSwap * numSwarms * sizeof(int));

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
			hPosition[i * numParticles + j] = (float) j;
			hVelocity[i * numParticles + j] = (float) j;
		}
	}



	//Copy the memory over to the GPU
	cudaMemcpy(dPosition, hPosition, numParticles * numSwarms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocity, hVelocity, numParticles * numSwarms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dBestSwapIndices, bestListing, numToSwap * numSwarms * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dWorstSwapIndices, worstListing, numToSwap * numSwarms * sizeof(int), cudaMemcpyHostToDevice);

	SwapBestParticles<<<numSwarms, numParticles>>>(numSwarms, numTasks, numToSwap, dBestSwapIndices, dWorstSwapIndices, dPosition, dVelocity);

	//Copy the data back
	cudaMemcpy(hPosition, dPosition, numParticles * numSwarms * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hVelocity, dVelocity, numParticles * numSwarms * sizeof(float), cudaMemcpyDeviceToHost);

	//Ensure that the correct modified numbers were added.
	for (i = 0; i < numSwarms; i++)
	{



	}



	PrintTestResults(passed);

	return passed;
}



void RunGPUCorrectnessTests()
{
	TestTextureReads();	
	TestTextureReadsRandom();
	TestGPUMakespan();
	TestDiscreteCoord();
	//TestSwapParticles();

}