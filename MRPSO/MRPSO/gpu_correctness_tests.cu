#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include "tests.h"
#include "helper.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

void PrintTestResults(int passed)
{
	if (passed)
		printf("[SUCCESS] Test passed!\n\n");
	else
		printf("[FAILURE] Test failed!\n\n");
}

__device__ int GetDiscreteCoordT(float val)
{
	return (int) rintf(val);
}

/* Unfortunately, we cannot do external calls to device code, so we have to copy this here under a DIFFERENT name(!!!)...
 * Thanks Nvidia!
 */
__device__ float CalcMakespanT(int numTasks, int numMachines, float *matching, float *scratch)
{
	int i;
	float makespan;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	makespan = 0.0f;

	//Clear our scratch table
	for (i = 0; i < numTasks; i++)
		scratch[(threadID * numMachines) + GetDiscreteCoordT(matching[threadID * numTasks + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		scratch[(threadID * numMachines) + GetDiscreteCoordT(matching[threadID * numTasks + i])] += tex2D(texETCMatrix, GetDiscreteCoordT(matching[threadID * numTasks + i]), i);

		if (scratch[(threadID * numMachines) + GetDiscreteCoordT(matching[threadID * numTasks + i])] > makespan)
			makespan = scratch[(threadID * numMachines) + GetDiscreteCoordT(matching[threadID * numTasks + i])];
	}	

	return makespan;
}

__global__ void TestTexture(int numTasks, int numMachines, float *outVals)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < numTasks * numMachines)
		outVals[idx] = tex2D(texETCMatrix, threadIdx.x, blockIdx.x);
}

__global__ void TestMakespan(int numTasks, int numMachines, int numMatchings, float *matching, float *scratch, float *outVal)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < numMatchings)
		outVal[threadID] = CalcMakespanT(numTasks, numMachines, matching, scratch);
}

bool TestTextureReads()
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
		hMatching[i] = rand() % GetNumMachines();

	//Compute the makespans on the CPU
	for (i = 0; i < numMatchings; i++)
		cpuMakespans[i] = ComputeMakespan(&hMatching[i * GetNumTasks()], GetNumTasks());

	cudaMalloc((void **)&dOut, sizeof(float) * numMatchings );
	cudaMalloc((void **)&matching, sizeof(float) * numMatchings * GetNumTasks() );
	cudaMalloc((void **)&scratch, sizeof(float) * numMatchings * GetNumMachines() );

	cudaMallocArray(&cuArray, &texETCMatrix.channelDesc, GetNumMachines(), GetNumTasks());
	cudaMemcpyToArray(cuArray, 0, 0, hETCMatrix, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texETCMatrix, cuArray);

	cudaMemcpy(matching, hMatching, sizeof(float) * numMatchings * GetNumTasks(), cudaMemcpyHostToDevice);
	cudaMemcpy(scratch, hScratch, sizeof(float) * numMatchings * GetNumMachines(), cudaMemcpyHostToDevice);

	texETCMatrix.normalized = false;
	texETCMatrix.filterMode = cudaFilterModePoint;

	TestMakespan<<<numBlocks, threadsPerBlock>>>(GetNumTasks(), GetNumMachines(), numMatchings, matching, scratch, dOut);

	cudaMemcpy(gpuMakespans, dOut, sizeof(float) * numMatchings, cudaMemcpyDeviceToHost);

	for (i = 0; i < numMatchings; i++)
	{
		if (abs(gpuMakespans[i] - cpuMakespans[i]) > ACCEPTED_DELTA)
		{
			printf("[ERROR] - GPU Makespan was: %f (expected: %f)\n", gpuMakespans[i], cpuMakespans[i]);
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


void RunGPUCorrectnessTests()
{
	TestTextureReads();
TestGPUMakespan();


}