#include <stdio.h>
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

/* InitializeParticles
 *
 * Initializes the position and velocity of the particles. Each thread is resposible
 * for a single dimension of a single particle.
 */
__global__ void InitializeParticles(int totalParticles, int numTasks, int numMachines, float *position, float *velocity, float *randNums)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float myRand1, myRand2;
	int randOffset;

	if (threadID < __mul24(totalParticles, numTasks))
	{
		randOffset = __mul24(totalParticles, numTasks);
		myRand1 = randNums[threadID];
		myRand2 = randNums[threadID + randOffset];
		position[threadID] = (numMachines - 1) * myRand1;	
		velocity[threadID] = (numMachines >> 1) * myRand2;
	}
}
__global__ void SwapBestParticles(int numSwarms, int numParticles, int numTasks, int numToSwap, int *bestSwapIndices, int *worstSwapIndices, float *position, float *velocity)
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
		mySwarmIndex = __mul24(mySwarm, numToSwap);
		neighborSwarmIndex = __mul24(neighbor, numToSwap);

		//The actual index within this swarm is the remainer of threadID / numToSwap
		myIndex = threadID % numToSwap;	

		//Compute the starting indices for the best and worst locations.
		bestIndex = mySwarm * numParticles * numTasks + __mul24(bestSwapIndices[mySwarmIndex + myIndex], numTasks);
		worstIndex = neighbor * numParticles * numTasks + __mul24(worstSwapIndices[neighborSwarmIndex + myIndex], numTasks);
		
		//Each of our threads that have work to do will swap all of the dimensions of 
		//one of the 'best' particles for its swarm with the corresponding 'worst'
		//particle of its neighboring swarm.
		for (i = 0; i < numTasks; i++)
		{
			//Store our velocities (the best values)
			tempPosition = position[bestIndex + i];
			tempVelocity = velocity[bestIndex + i];

			//Swap the other swarm's worst into our best
			position[bestIndex + i] = position[worstIndex + i];
			velocity[bestIndex + i] = velocity[worstIndex + i];

			//Finally swap our best values into the other swarm's worst
			position[worstIndex + i] = tempPosition;
			velocity[worstIndex + i] = tempVelocity;
		}
	}
}

__device__ float ClampVelocity(int numMachines, float velocity)
{
	float clamp = 0.5f * numMachines;

	if (velocity > clamp)
		velocity = clamp;
	else if (velocity < -clamp)
		velocity = -clamp;

	return velocity;
}

__device__ float ClampPosition(int numMachines, float position)
{
	if (position < 0.0f)
		position = 0.0f;
	else if (position > numMachines - 1)
		position = (float) numMachines - 1;

	return position;
}

__global__ void UpdateVelocityAndPosition(int numSwarms, int numParticles, int numMachines, int numTasks, int iterationNum, float *velocity, float *position, 
										  float *pBestPosition, float *gBestPosition, float *rands, ArgStruct args)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float newVel;
	float currPos;
	int randOffset;
	int totalParticles = numSwarms * numParticles;
	int gBestOffset;

	//Each thread is responsible for updating one dimension of one particle's 
	if (threadID < __mul24(totalParticles, numTasks))
	{
		//Two separate random numbers for every dimension for each particle each iteration.
		randOffset = totalParticles * numTasks * iterationNum * 2 + (threadID * 2);

		//The swarm this particle belongs to simply the number of threads handling each swarm (numParticles * numTasks)
		//divided by this thread's threadID.
		gBestOffset = (threadID / (numParticles * numTasks)) * numTasks;
		gBestOffset += threadID % numTasks;

		currPos = position[threadID];
		newVel = velocity[threadID];

		newVel *= args.x;		
		newVel += args.z * rands[randOffset] * (pBestPosition[threadID] - currPos);
		newVel += args.w * rands[randOffset + 1] * (gBestPosition[gBestOffset] - currPos);	

		//Write out our velocity
		newVel = ClampVelocity(numMachines, newVel);
		velocity[threadID] = newVel;

		//Update the position
		currPos += newVel;
		currPos = ClampPosition(numMachines, currPos);
		position[threadID] = currPos;
	}
}

__global__ void UpdateBests(int numSwarms, int numParticles, int numTasks, float *position, float *pBest, float *pBestPosition)
{



}


float *MRPSODriver(RunConfiguration *run)
{
	int i;
	float *matching = NULL;
	int threadsPerBlock, numBlocks;
	int totalComponents;
	int numMachines, numTasks;
	ArgStruct args;

	threadsPerBlock = run->threadsPerBlock;
	totalComponents = run->numSwarms * run->numParticles * numTasks;
	args.x = run->w;
	args.y = run->wDecay;
	args.z = run->c1;
	args.w = run->c2;

	numMachines = GetNumMachines();
	numTasks = GetNumTasks();

	numBlocks = CalcNumBlocks(totalComponents, threadsPerBlock);

	//Initialize our particles.
	InitializeParticles<<<numBlocks, threadsPerBlock>>>(run->numSwarms * run->numParticles, numTasks, numMachines, dPosition, dVelocity, dRands);

	//Run MRPSO GPU for the given number of iterations.
	for (i = 1; i <= run->numIterations; i++)
	{
		//Update the Position and Velocity
		UpdateVelocityAndPosition<<<numBlocks, threadsPerBlock>>>(run->numSwarms, run->numParticles, numMachines, numTasks, i - 1, 
																  dVelocity, dPosition, dPBestPosition, dGBestPosition, dRands, args);

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

