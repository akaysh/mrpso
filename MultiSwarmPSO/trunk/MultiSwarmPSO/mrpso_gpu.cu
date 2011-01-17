#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "curand.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

extern __shared__ float sharedScratch[];

__device__ float CalcMakespanShared(int numTasks, int numMachines, float *matching)
{
	int i;
	float makespan;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int taskOffset;
	float matchingVal;
	float val;
	
	makespan = 0.0f;
	taskOffset = __mul24(threadID, numTasks);

	//Clear our scratch table
	for (i = 0; i < numTasks; i++)
		sharedScratch[(int) floorf(matching[taskOffset + i])] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		matchingVal = matching[taskOffset + i];

		sharedScratch[(int) floorf(matchingVal)] += tex2D(texETCMatrix, matchingVal, (float) i);
		val = sharedScratch[(int) floorf(matchingVal)];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

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

__global__ void UpdateFitness(int numSwarms, int numParticles, int numTasks, int numMachines, float *position, float *scratch, float *fitness)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < __mul24(numSwarms, numParticles))
		fitness[threadID] = CalcMakespan(numTasks, numMachines, position, scratch);
}

/* UpdateBests
 * 
 * Updates both the particle-bests and swarm-best values.
 * Each block must contains enough threads to handle each particle in a swarm.
 * Shared memory requirements are the number of particles in a swarm * 2
 */
__global__ void UpdateBests(int numSwarms, int numParticles, int numTasks, float *pBest, float *pBestPositions, float *gBest, float *gBestPositions,
							float *position, float *fitness)
{
	extern __shared__ float fitnessValues[];
	__shared__ float *indexValues;
	int i;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int updateFitness;
	int gBestIndex;

	indexValues = &fitnessValues[blockDim.x];

	//Push the fitness values for this swarm into shared memory
	if (threadIdx.x < numParticles)
	{	
		fitnessValues[threadIdx.x] = fitness[threadID];
		indexValues[threadIdx.x] = threadID * numTasks;
	}

	//Each thread determines if they need to update their own pbest value.
	//If so, each thread updates the pBest and pBestPosition for their own data.
	if (fitnessValues[threadIdx.x] < pBest[threadID])
	{
		pBest[threadID] = fitnessValues[threadIdx.x];
		
		for (i = 0; i < numTasks; i++)
		{
			pBestPositions[threadID * numTasks + i] = position[threadID * numTasks + i];
		}
	}

	__syncthreads();

	//Parallel reduction to find best fitness amongst threads in swarm
	//We do this reduction in shared memory.
	for (i = blockDim.x / 2; i > 0; i >>= 1)
	{
		if (threadIdx.x < i)
		{
			if (fitnessValues[threadIdx.x] > fitnessValues[threadIdx.x + i])
			{				
				fitnessValues[threadIdx.x] = fitnessValues[threadIdx.x + i];
				indexValues[threadIdx.x] = indexValues[threadIdx.x + i];
			}
		}
		__syncthreads();
	}

	//All threads check if gBest must be updated (we just do this to avoid collaboration)
	//Both the shared and global memory values will be broadcast anyways as each thread is
	//accessing the same value, so the performance loss will be minimal.
	updateFitness = 0;	
	__syncthreads();

	if (fitnessValues[0] < gBest[blockIdx.x])
	{
		
		updateFitness = 1;	
	}

	//Update gBest and gBestPosition by using all threads in a for loop if necessary
	if (updateFitness)
	{
		for (i = threadIdx.x; i < numTasks; i += blockDim.x)
		{
			gBestPositions[blockIdx.x * numTasks + i] = pBestPositions[(int) indexValues[0] + i];			
		}

		if (threadIdx.x == 0)
			gBest[blockIdx.x] = fitnessValues[0];
	}
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


float *MRPSODriver(RunConfiguration *run)
{
	int i;
	float *matching = NULL;
	int threadsPerBlock, numBlocks, numBlocksFitness;
	int fitnessRequired;
	int totalComponents;
	int numMachines, numTasks;
	ArgStruct args;

	threadsPerBlock = run->threadsPerBlock;
	totalComponents = run->numSwarms * run->numParticles * numTasks;
	fitnessRequired = run->numSwarms * run->numParticles;
	args.x = run->w;
	args.y = run->wDecay;
	args.z = run->c1;
	args.w = run->c2;

	numMachines = GetNumMachines();
	numTasks = GetNumTasks();

	numBlocks = CalcNumBlocks(totalComponents, threadsPerBlock);
	numBlocksFitness = CalcNumBlocks(fitnessRequired, 128);

	//Initialize our particles.
	InitializeParticles<<<numBlocks, threadsPerBlock>>>(run->numSwarms * run->numParticles, numTasks, numMachines, dPosition, dVelocity, dRands);

	//Update the Fitness for the first time...
	UpdateFitness<<<numBlocksFitness, 128>>>(run->numSwarms, run->numParticles, numTasks, numMachines, dPosition, dScratch, dFitness);

	//Run MRPSO GPU for the given number of iterations.
	for (i = 1; i <= run->numIterations; i++)
	{
		//Update the Position and Velocity
		UpdateVelocityAndPosition<<<numBlocks, threadsPerBlock>>>(run->numSwarms, run->numParticles, numMachines, numTasks, i - 1, 
																  dVelocity, dPosition, dPBestPosition, dGBestPosition, dRands, args);
		//Update the Fitness
		UpdateFitness<<<numBlocksFitness, 128>>>(run->numSwarms, run->numParticles, numTasks, numMachines, dPosition, dScratch, dFitness);

		//Update the local and swarm best positions
		UpdateBests<<<run->numSwarms, run->numParticles>>>(run->numSwarms, run->numParticles, numTasks, dPBest, dPBestPosition, dGBest, dGBestPosition,
														   dPosition, dFitness);

		if (i % run->iterationsBeforeSwap == 0)
		{
			//Build up the swap indices for each swarm


			//Swap particles between swarms
			SwapBestParticles<<<1, 1024>>>(run->numSwarms, run->numParticles, numTasks, run->numParticlesToSwap, dBestSwapIndices, 
				                           dWorstSwapIndices, dPosition, dVelocity);


		}
	}

	return matching;
}

