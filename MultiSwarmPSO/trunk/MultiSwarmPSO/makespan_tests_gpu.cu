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

	printf("\tRunning GPU Makespan Test...\n");

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
	cudaThreadSynchronize();

	cudaMemcpy(gpuMakespans, dOut, sizeof(float) * numMatchings , cudaMemcpyDeviceToHost);

	for (i = 0; i < numMatchings; i++)
	{
		if (abs(gpuMakespans[i] - cpuMakespans[i]) > ACCEPTED_DELTA)
		{
			printf("\t[ERROR] - %d GPU Makespan was: %f (expected: %f)\n", i, gpuMakespans[i], cpuMakespans[i]);
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

	printf("\nStarting GPU makespan tests...\n\n");

	passed &= TestGPUMakespan();

	if (passed)
		printf("[PASSED] All makespan tests passed!\n\n");
	else
		printf("[FAILED] makespan tests failed!\n\n");
}