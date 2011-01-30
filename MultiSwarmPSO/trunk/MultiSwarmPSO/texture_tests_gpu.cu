#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "tests.h"
#include "helper.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

__device__ int GetDiscreteCoordT(float val)
{
	return  floorf(val);
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

int TestTextureReads()
{
	int i;
	int passed = 1;
	cudaArray *cuArray;
	float *dOut;
	float *gpuETCMatrix;

	printf("\tRunning Texture Read Test...\n");
	
	BuildMachineList("machines100.txt");
	BuildTaskList("tasks1000.txt");
	GenerateETCMatrix();

	gpuETCMatrix = (float *) malloc(GetNumMachines() * GetNumTasks() * sizeof(float));
	cudaMalloc((void **)&dOut, GetNumMachines() * GetNumTasks() * sizeof(float));

	cudaMallocArray(&cuArray, &texETCMatrix.channelDesc, GetNumMachines(), GetNumTasks());
	cudaMemcpyToArray(cuArray, 0, 0, hETCMatrix, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texETCMatrix, cuArray);

	texETCMatrix.normalized = false;
	texETCMatrix.filterMode = cudaFilterModePoint;

	TestTexture<<<1000, 100>>>(GetNumTasks(), GetNumMachines(), dOut);
	cudaThreadSynchronize();

	cudaMemcpy(gpuETCMatrix, dOut, sizeof(float)*GetNumMachines() *GetNumTasks(), cudaMemcpyDeviceToHost);

	for (i = 0; i < GetNumTasks() * GetNumMachines(); i++)
	{
		if (gpuETCMatrix[i] - hETCMatrix[i] > ACCEPTED_DELTA)
		{
			printf("\t[ERROR] - GPU ETC Matrix was: %f (expected: %f)\n", gpuETCMatrix[i], hETCMatrix[i]);
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

	printf("\tRunning Texture Read Random Test...\n");
	
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
	cudaThreadSynchronize();

	cudaMemcpy(gpuOut, dOut, GetNumTasks() * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < GetNumTasks(); i++)
		cpuOut[i] = hETCMatrix[(i * GetNumMachines()) + DiscreteCoord(hMatching[i])];

	for (i = 0; i < GetNumTasks(); i++)
	{
		if (abs(gpuOut[i] - cpuOut[i]) > ACCEPTED_DELTA)
		{
			printf("\t[ERROR] - %d GPU ETC Matrix was: %f (expected: %f)\n", i, gpuOut[i], cpuOut[i]);
			printf("\t\tOriginal matching value used: %f\n", hMatching[i]);
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

void RunGPUTextureTests()
{
	int passed = 1;

	printf("\nStarting GPU Texture tests...\n\n");

	passed &= TestTextureReads();
	passed &= TestTextureReadsRandom();

	if (passed)
		printf("[PASSED] All texture tests passed!\n\n");
	else
		printf("[FAILED] Texture tests failed!\n\n");
}
