#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"

texture<float, 2, cudaReadModeElementType> TexETCMatrix;

__device__ int GetDiscreteCoord(float val)
{
	return (int) rintf(val);
}

__device__ float CalcMakespan(int numTasks, int numMachines, float *matching, float *scratch)
{
	int i;
	float makespan;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	makespan = 0.0f;

	//Clear our scratch table
	for (i = 0; i < numTasks; i++)
		scratch[(threadID * numMachines) + GetDiscreteCoord(matching[threadID * numTasks + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		scratch[(threadID * numMachines) + GetDiscreteCoord(matching[threadID * numTasks + i])] += tex2D(TexETCMatrix, GetDiscreteCoord(matching[threadID * numTasks + i]), i);

		if (scratch[(threadID * numMachines) + GetDiscreteCoord(matching[threadID * numTasks + i])] > makespan)
			makespan = scratch[(threadID * numMachines) + GetDiscreteCoord(matching[threadID * numTasks + i])];
	}	

	return makespan;
}

__global__ void SwapBestParticles(int numSwarms, int numParticles, int *swapIndices, float *position, float *velocity, float *fitness, float *newPosition, float *newVelocity, float *newFitness)
{
	//Perform the swap into the new_____ arrays (double buffering)


}

__global__ void RunIteration(int numSwarms, int numParticles, float *ETCMatrix, float *position, float *velocity, float *pBest, float *pBestPosition, float *gBest, float *gBestPosition)
{
	extern __shared__ float sharedPBest[]; //Local best positions are stored in shared memory 
										   //and dumped to global memory at the end of execution of this kernel.
	float fitness; //Fitness values are stored in registers as we do not need these values to persist.
	


}


float *MRPSODriver()
{
	float *matching = NULL;

	return matching;
}


