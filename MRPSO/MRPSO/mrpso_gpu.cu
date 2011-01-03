#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

__device__ float CalcMakespan(int numTasks, int numMachines, float *matching, float *scratch)
{
	int i;
	float makespan;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int taskOffset, machineOffset;
	float val;
	
	makespan = 0.0f;
	taskOffset = __mul24(threadID, numTasks);
	machineOffset = __mul24(threadID, numMachines);

	//Clear our scratch table
	for (i = 0; i < numTasks; i++)
		scratch[machineOffset + (int) floorf(matching[taskOffset + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		scratch[machineOffset + (int) floorf(matching[taskOffset + i])] += tex2D(texETCMatrix, matching[taskOffset + i], (float) i);
		val = scratch[machineOffset + (int) floorf(matching[taskOffset + i])];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

__global__ void SwapBestParticles(int numSwarms, int numParticles, int *swapIndices, float *position, float *velocity, float *fitness, float *newPosition, float *newVelocity, float *newFitness)
{
	//Perform the swap into the new_____ arrays (double buffering)


}

__global__ void RunIteration(int numSwarms, int numParticles, float *position, float *velocity, float *pBest, float *pBestPosition, float *gBest, float *gBestPosition)
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


