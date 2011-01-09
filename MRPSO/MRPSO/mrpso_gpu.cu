#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "curand.h"

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

__global__ void SwapBestParticles(int numSwarms, int numTasks, int numToSwap, int *swapIndices, float *position, float *velocity)
{
	int i;
	int bestIndex, worstIndex;
	int neighbor;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float tempPosition, tempVelocity;
	int mySwarm, myIndex;
	int mySwarmIndex, neighborSwarmIndex;

	if (threadID < __mul24(numSwarms, numToSwap))
	{
		//Determine which swarm this thread is covering.
		mySwarm = threadID / numToSwap;
		neighbor = mySwarm < numSwarms - 1 ? mySwarm + 1 : 0;
		mySwarmIndex = __mul24(mySwarm, (numToSwap << 1));
		neighborSwarmIndex = __mul24(neighbor, (numToSwap << 1));

		//The actual index within this swarm is the remainer of threadID / numToSwap
		myIndex = threadID % numToSwap;	

		//Compute the starting indices for the best and worst locations.
		bestIndex = swapIndices[mySwarmIndex + myIndex];
		worstIndex = swapIndices[neighborSwarmIndex + myIndex];
		
		//Each of our threads that have work to do will swap all of the dimensions of 
		//one of the 'best' particles for its swarm with the corresponding 'worst'
		//particle of its neighboring swarm.
		for (i = 0; i < numTasks; i++)
		{
			//Store our velocities (the best values)
			tempPosition = position[bestIndex + i];
			tempVelocity = velocity[bestIndex + i];

			//Swap the other swarm's worst into our best
			position[bestIndex] = position[worstIndex + i];
			velocity[bestIndex] = velocity[worstIndex + i];

			//Finally swap our best values into the other swarm's worst
			position[worstIndex + i] = tempPosition;
			velocity[worstIndex + i] = tempVelocity;
		}
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

	//Have the first numTasks threads load in the gBestPosition into shared memory so it can be broadcast while updating velocity.
	if (threadIdx.x < numTasks)
	{
		sharedGBestPosition[threadIdx.x] = gBestPosition[blockIdx.x * numTasks + threadIdx.x];
	}

	__syncthreads();

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

__global__ void UpdateVelocity(int numSwarms, int numParticles, int numTasks, float *position, 
							  float *velocity, float *pBest, float *pBestPosition, float *gBest, float *gBestPosition, float *rands, ArgStruct args)
{
	float fitness; //Fitness values are stored in registers as we do not need these values to persist.
	int i;


	//Update fitness

	//Update local best

	__syncthreads();

	//Update global best
		
	__syncthreads();


}


float *MRPSODriver(RunConfiguration *run)
{
	int i;
	float *matching = NULL;

	//Run MRPSO GPU for the given number of iterations.
	for (i = 1; i <= run->numIterations; i++)
	{
		//Update the Position and Velocity

		//Update the Fitness

		//Update the local and swarm best positions

		if (i % run->iterationsBeforeSwap == 0)
		{
			//Build up the swap indices for each swarm


			//Swap particles between swarms

		}


	}

	return matching;
}


