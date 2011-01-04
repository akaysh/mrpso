#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

extern __shared__ float sharedGBestPosition[];

__device__ float CalcMakespan(int numTasks, int numMachines, float *matching, float *scratch)
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
		scratch[machineOffset + (int) floorf(matching[taskOffset + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		matchingVal = matching[taskOffset + i];

		scratch[machineOffset + (int) floorf(matchingVal)] += tex2D(texETCMatrix, matchingVal, (float) i);
		val = scratch[machineOffset + (int) floorf(matchingVal)];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

__device__ void UpdateMakespan(int numParticles, int numTasks, int numMachines, float *position, float *scratch)
{



}

__global__ void SwapBestParticles(int numSwarms, int numParticles, int numToSwap, int *swapIndices, float *position, float *velocity)
{
	int i;
	int bestIndex, worstIndex;
	int neighbor;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float tempPosition, tempVelocity;

	neighbor = blockIdx.x < numSwarms - 1 ? blockIdx.x + 1 : 0;

	//swapIndices contains entries in the form:
	// P1S1_best1, P2S1_best1, ..., P1S1_best2, ...P1S1_worst1, ...PnS1_worstn
	// P1S2_best1,...
	for (i = 0; i < numToSwap; i++)
	{
		                          //			Swarm index						  Particle best index for iteration
		bestIndex = swapIndices[ (blockIdx.x * (numParticles * numToSwap * 2)) + ((numParticles * i) + threadIdx.x)];

		worstIndex = swapIndices[(neighbor * (numParticles * numToSwap * 2)) + ((numParticles * i * 2) + threadIdx.x)];

		//Store our velocities (the best values)
		tempPosition = position[bestIndex];
		tempVelocity = velocity[bestIndex];

		//Swap the other swarm's worst into our best
		position[bestIndex] = position[worstIndex];
		velocity[bestIndex] = velocity[worstIndex];

		//Finally swap our best values into the other swarm's worst
		position[worstIndex] = tempPosition;
		velocity[worstIndex] = tempVelocity;
	}
}

__device__ void UpdateVelocityAndPosition(int numParticles, int numTasks, float *velocity, float *position, float *pBestPosition, 
										  float *gBestPosition, float *rands, ArgStruct args)
{
	//pBest offset is our block's offset + our thread id within the block
	//Our block's offset is the block number times the number of particles per swarm times the dimenions (# of tasks).
	int offset = (blockIdx.x * numParticles * numTasks) + (threadIdx.x * numTasks);
	float newVel;
	float lperb, gperb;
	float currPos;
	int i;

	for (i = 0; i < numTasks; i++)
	{
		currPos = position[offset + i];
		newVel = velocity[offset + i];
		
		newVel *= args.x;
		lperb = args.z * rands[(blockIdx.x * numParticles * numTasks * 2) + (threadIdx.x * 2)] * (pBestPosition[offset + i] - currPos);
		gperb = args.w * rands[(blockIdx.x * numParticles * numTasks * 2) + (threadIdx.x * 2 + 1)] * (sharedGBestPosition[i] - currPos);

		newVel += lperb + gperb;
		velocity[offset + i] = newVel;

		//Might as well update the position along this dimension while we're at it.
		position[offset + i] += newVel;
	}
}

__global__ void RunIterations(int numSwarms, int numParticles, int numTasks, int itersToRun, float *position, float *velocity, float *pBest, float *pBestPosition, float *gBest, float *gBestPosition)
{
	extern __shared__ float sharedPBest[]; //Local best positions are stored in shared memory 
										   //and dumped to global memory at the end of execution of this kernel.
	float fitness; //Fitness values are stored in registers as we do not need these values to persist.
	int i;

	//Have the first numTasks threads load in the gBestPosition into shared memory so it can be broadcast while updating velocity.
	if (threadIdx.x < numTasks)
	{
		sharedGBestPosition[threadIdx.x] = gBestPosition[blockIdx.x * numTasks + threadIdx.x];
	}

	__syncthreads();

	for (i = 0; i < itersToRun; i++)
	{

		//Update velocity and position

		//Update fitness

		//Update local best

		__syncthreads();

		//Update global best
		
		__syncthreads();
	}


}


float *MRPSODriver()
{
	float *matching = NULL;

	return matching;
}


