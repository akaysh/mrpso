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
	int myRand1, myRand2, randOffset;

	if (threadID < __mul24(totalParticles, numTasks))
	{
		randOffset = __mul24(totalParticles, numTasks);
		myRand1 = randNums[threadID];
		myRand2 = randNums[threadID + randOffset];
		position[threadID] = __mul24(numMachines - 1, myRand1);	
		velocity[threadID] = __mul24(numMachines >> 2, myRand2);
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
	else if (position > numMachines)
		position = (float) numMachines;

	return position;
}

__device__ void UpdateVelocityAndPosition(int numSwarms, int numParticles, int numMachines, int numTasks, float *velocity, float *position, float *pBestPosition, 
										  float *gBestPosition, float *rands, ArgStruct args)
{
	//Positions, Velocities, and pBests are stored as:
	// s1p1v1, s1p2v1, s1p3v1, ..., pnvn
	// s2p1v1, ...
	int swarmOffset = blockIdx.x * numParticles * numTasks;
	float newVel;
	float lperb, gperb;
	float currPos;
	int i;

	//Push the global best position into shared memory so these values can be broadcast to all threads later.
	for (i = threadIdx.x; i < numTasks; i+= blockDim.x)
	{
		sharedGBestPosition[i] = gBestPosition[blockIdx.x * numTasks + i];
	}

	__syncthreads();

	for (i = 0; i < numTasks; i++)
	{
		currPos = position[swarmOffset + (i * numParticles) + threadIdx.x];
		newVel = velocity[swarmOffset + (i * numParticles) + threadIdx.x];
		
		newVel *= args.x;
		lperb = args.z * rands[(blockIdx.x * numParticles * numTasks * 2) + (threadIdx.x * 2)] * (pBestPosition[swarmOffset + (i * numParticles) + threadIdx.x] - currPos);
		gperb = args.w * rands[(blockIdx.x * numParticles * numTasks * 2) + (threadIdx.x * 2 + 1)] * (sharedGBestPosition[i] - currPos);

		newVel += lperb + gperb;

		//Clamp the velocity if required.
		newVel = ClampVelocity(numMachines, newVel);

		//Write out our velocity to global memory.
		velocity[swarmOffset + (i * numParticles) + threadIdx.x] = newVel;

		//Might as well update the position along this dimension while we're at it.
		currPos += newVel;
		currPos = ClampPosition(numTasks, currPos);
		position[swarmOffset + (i * numParticles) + threadIdx.x] = currPos;
	}
}

__global__ void UpdateBests(int numSwarms, int numParticles, int numTasks, float *position, float *pBest, float *pBestPosition)
{



}

__global__ void RunIteration(int numSwarms, int numParticles, int numMachines, int numTasks, int totalIters, float *position, float *velocity, float *pBest, 
							 float *pBestPosition, float *gBest, float *gBestPosition, float *bestSwap, float *worstSwap, float *rands, ArgStruct args)
{
	float fitness; //Fitness values are stored in registers as we do not need these values to persist.
	int i;

	for (i = 0; i < totalIters; i++)
	{
		//Update velocity and position of the particle.
		UpdateVelocityAndPosition(numSwarms, numParticles, numMachines, numTasks, velocity, position, pBestPosition, gBestPosition, rands, args);

		//Update the local and global best values (we implicitly compute the fitness here).

		__syncthreads();

		//Update global best
			
		__syncthreads();
	}
}


float *MRPSODriver(RunConfiguration *run)
{
	int i;
	float *matching = NULL;

	return matching;
}

float *MRPSODriverIndividual(RunConfiguration *run)
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

