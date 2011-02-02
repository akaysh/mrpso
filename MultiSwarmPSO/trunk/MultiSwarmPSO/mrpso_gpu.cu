#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper.h"
#include "curand.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;
cudaArray *cuArray;

extern __shared__ float sharedScratch[];

__device__ float CalcMakespanShared(int numParticles, int numTasks, int numMachines, float *matching)
{
	int i;
	int swarmOffset, scratchOffset;
	float makespan;
	float matchingVal;
	float val;
	
	//The position values that this thread needs to retrieve are located at its swarm offset (mySwarm * numParticles * numTasks)
	//and we add i * numParticles as the offset at each point in the for loop.
	swarmOffset = blockIdx.x * numParticles * numTasks;
	scratchOffset = threadIdx.x * numMachines;

	makespan = 0.0f;

	//Clear our scratch table+
	for (i = 0; i < numMachines; i++)
		sharedScratch[scratchOffset + i] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		matchingVal = matching[swarmOffset + (i * numParticles) + threadIdx.x];
		sharedScratch[scratchOffset + (int) floorf(matchingVal)] += tex2D(texETCMatrix, matchingVal, (float) i);
		val = sharedScratch[scratchOffset + (int) floorf(matchingVal)];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

__device__ float CalcMakespan(int numParticles, int numTasks, int numMachines, float *matching, float *scratch)
{
	int i;
	int swarmOffset, scratchOffset;
	float makespan;
	float matchingVal;
	float val;
	
	//The position values that this thread needs to retrieve are located at its swarm offset (mySwarm * numParticles * numTasks)
	//and we add i * numParticles as the offset at each point in the for loop.
	swarmOffset = blockIdx.x * numParticles * numTasks;
	scratchOffset = blockIdx.x * numParticles * numMachines;

	makespan = 0.0f;

	//Clear our scratch table+
	for (i = 0; i < numMachines; i++)
		scratch[scratchOffset + (i * numParticles) + threadIdx.x] = 0.0f;

	for (i = 0; i < numTasks; i++)
	{
		matchingVal = matching[swarmOffset + (i * numParticles) + threadIdx.x];
		scratch[scratchOffset + (int) (floorf(matchingVal) * numParticles) + threadIdx.x] += tex2D(texETCMatrix, matchingVal, (float) i);
		val = scratch[scratchOffset + (int) (floorf(matchingVal) * numParticles) + threadIdx.x];

		if (val > makespan)
			makespan = val;
	}	

	return makespan;
}

__global__ void UpdateFitness(int numSwarms, int numParticles, int numTasks, int numMachines, float *position, float *scratch, float *fitness)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < __mul24(numSwarms, numParticles))
		fitness[threadID] = CalcMakespan(numParticles, numTasks, numMachines, position, scratch);
}

__global__ void UpdateFitnessShared(int numSwarms, int numParticles, int numTasks, int numMachines, float *position, float *fitness)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (threadID < __mul24(numSwarms, numParticles))
		fitness[threadID] = CalcMakespanShared(numParticles, numTasks, numMachines, position);
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
	int index;

	indexValues = &fitnessValues[blockDim.x];

	//Push the fitness values for this swarm into shared memory
	if (threadIdx.x < numParticles)
	{	
		fitnessValues[threadIdx.x] = fitness[threadID];
		indexValues[threadIdx.x] = threadIdx.x;
	}

	//Each thread determines if they need to update their own pbest value.
	//If so, each thread updates the pBest and pBestPosition for their own data.
	if (fitnessValues[threadIdx.x] < pBest[threadID])
	{
		pBest[threadID] = fitnessValues[threadIdx.x];

		index = blockIdx.x * numParticles * numTasks + threadIdx.x;
		
		for (i = 0; i < numTasks; i++)
		{
			pBestPositions[index + (i * numParticles)] = position[index + (i * numParticles)];
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
	
	if (fitnessValues[0] < gBest[blockIdx.x])
	{		
		updateFitness = 1;	
	}

	//Update gBest and gBestPosition by using all threads in a for loop if necessary
	if (updateFitness)
	{
		index = blockIdx.x * numParticles * numTasks  + (int) indexValues[0]; 

		for (i = threadIdx.x; i < numTasks; i += blockDim.x)
		{
			gBestPositions[blockIdx.x * numTasks + i] = pBestPositions[index + (i * numParticles)];			
		}

		if (threadIdx.x == 0)
		{
			//printf("Found global  best value: %f\n", fitnessValues[0]);
			gBest[blockIdx.x] = fitnessValues[0];
		}
	}
}

/* InitializeParticles
 *
 * Initializes the position and velocity of the particles. Each thread is resposible
 * for a single dimension of a single particle.
 */
__global__ void InitializeParticles(int numSwarms, int numParticles, int numTasks, int numMachines, float *gBests, float *pBests, float *position, 
									float *velocity, float *randNums)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float myRand1, myRand2;
	int randOffset;
	int totalParticles = __mul24(numSwarms, numParticles);

	if (threadID < __mul24(totalParticles, numTasks))
	{
		randOffset = __mul24(totalParticles, numTasks);
		myRand1 = randNums[threadID];
		myRand2 = randNums[threadID + randOffset];
		position[threadID] = (numMachines - 1) * myRand1;	
		velocity[threadID] = (numMachines >> 1) * myRand2;
		
		if (threadID < totalParticles)
		{
			pBests[threadID] = 99999999.99f;

			if (threadID < numSwarms)
			{
				gBests[threadID] = 999999999.99f;
			}
		}
	}
}


__global__ void SwapBestParticles(int numSwarms, int numParticles, int numTasks, int numToSwap, int *bestSwapIndices, int *worstSwapIndices, 
								  float *position, float *velocity, float *pBest, float *pBestPosition)
{
	int bestIndex, worstIndex;
	int neighbor;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float tempPosition, tempVelocity, tempPBestPosition;
	int mySwarm, mySwapIndex, neighborSwapIndex;
	int mySwapIndicesBase, neighborSwapIndicesBase;
	int myDimension;

	if (threadID < numSwarms * numToSwap * numTasks)
	{	
		//First, figure out what swarm we are covering and who our neighbor is...
		mySwarm = threadID / (numToSwap * numTasks);
		neighbor = mySwarm < numSwarms - 1? mySwarm + 1 : 0;

		//Now, figure out what our starting index is into the swap indices (numToSwap records for each swarm)
		mySwapIndicesBase = mySwarm * (numToSwap);
		neighborSwapIndicesBase = neighbor * (numToSwap);		

		//Now let's figure out which actual swap within this swap we're responsible for as there's numToSwap choices!
		//And, while we're at it, figure out what dimension we're covering.
		mySwapIndex = (threadID / numTasks) % numToSwap;
		neighborSwapIndex = neighbor > 0 ? mySwapIndex : (threadID / numTasks) % numToSwap;
		myDimension = (threadID % numTasks);
		//printf("thread id %d dimension is %d\n", threadID, myDimension);

		//printf("t %d has myswap %d, neig %d with mySwapIndex of %d\n", threadID, mySwapIndicesBase, neighborSwapIndicesBase, mySwapIndex);

		//Finally let's get our indices!!
		bestIndex = (mySwarm * numParticles * numTasks) + myDimension * numParticles + bestSwapIndices[mySwapIndicesBase + mySwapIndex];
		worstIndex = (neighbor * numParticles * numTasks) + myDimension * numParticles + worstSwapIndices[neighborSwapIndicesBase + neighborSwapIndex];

//printf("Thread %d is choosing swaps from %d for best and %d for worst\n", threadID, mySwapIndicesBase + mySwapIndex, neighborSwapIndicesBase + neighborSwapIndex);
//printf("Thread %d will be taking from %d and putting in %d\n", threadID, bestIndex + myDimension, worstIndex + myDimension);


		//Store the best positions temporarily.
		tempPosition = position[bestIndex];
		tempVelocity = velocity[bestIndex];
		tempPBestPosition = pBestPosition[bestIndex];

		//Swap the other swarm's worst into our best
		position[bestIndex] = position[worstIndex];
		velocity[bestIndex] = velocity[worstIndex];
		pBestPosition[bestIndex] = pBestPosition[worstIndex];

		//Finally swap our best values into the other swarm's worst
		position[worstIndex] = tempPosition;
		velocity[worstIndex] = tempVelocity;
		pBestPosition[worstIndex] = tempPBestPosition;

		//Update the pBest value...
		if (threadID < numSwarms * numToSwap)
		{
			mySwarm = threadID / numToSwap;
			neighbor = mySwarm < numSwarms - 1 ? mySwarm + 1 : 0;
			mySwapIndex = threadID % numToSwap;

			bestIndex = mySwarm * numParticles + bestSwapIndices[mySwarm * numToSwap + mySwapIndex];
			worstIndex = neighbor * numParticles + worstSwapIndices[neighbor * numToSwap + mySwapIndex];

			//printf("Thread %d choosing from swap index %d for best and %d for worst\n", threadID, mySwarm * numToSwap + mySwapIndex, neighbor * numToSwap + mySwapIndex);

			tempPosition = pBest[bestIndex];
			pBest[bestIndex] = pBest[worstIndex];
			pBest[worstIndex] = tempPosition;
		}
		
	}
}

/* GenerateSwapIndices
 *
 * Generates the swap indices for swapping particles. Finds the best numToSwap and
 * the worst numToSwap particles from each swarm and records the values.
 *
 * @BLOCKDIM - Requires numParticles particles per thread block.
 * @SHAREDMEM - Requires numParticles * 5 + numToSwap * 2 elements of shared memory.
 */
__global__ void GenerateSwapIndices(int numSwarms, int numParticles, int numToSwap, float *fitness, int *bestSwapIndices, int *worstSwapIndices)
{
	extern __shared__ float sharedFitnessOriginal[];
	__shared__ float *sharedFitnessBest, *sharedFitnessWorst;
	__shared__ float *sharedIndicesBest, *sharedIndicesWorst;
	__shared__ float *sharedTempIndicesBest, *sharedTempIndicesWorst;
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j;

	sharedFitnessBest = &sharedFitnessOriginal[blockDim.x];	
	sharedFitnessWorst = &sharedFitnessBest[blockDim.x];
	sharedTempIndicesBest = &sharedFitnessWorst[blockDim.x];
	sharedTempIndicesWorst = &sharedTempIndicesBest[blockDim.x];	
	sharedIndicesBest = &sharedTempIndicesWorst[blockDim.x];
	sharedIndicesWorst = &sharedIndicesBest[numToSwap];	

	//Push the fitness values for this swarm into shared memory
	if (threadIdx.x < numParticles)
	{	
		sharedFitnessOriginal[threadIdx.x] = fitness[threadID];
		sharedFitnessBest[threadIdx.x] = sharedFitnessOriginal[threadIdx.x];
		sharedFitnessWorst[threadIdx.x] = sharedFitnessBest[threadIdx.x];
		sharedTempIndicesBest[threadIdx.x] = threadIdx.x;
		sharedTempIndicesWorst[threadIdx.x] = threadIdx.x;
	}

	//Main loop to find the best/worst particles.
	for (i = 0; i < numToSwap; i++)
	{		
		for (j = blockDim.x / 2; j > 0; j >>= 1)
		{			
			if (threadIdx.x < j)
			{
				if (sharedFitnessBest[threadIdx.x] == -1 ||
					(sharedFitnessBest[threadIdx.x] > sharedFitnessBest[threadIdx.x + j] && sharedFitnessBest[threadIdx.x + j] != -1))
				{
					//printf("\t[[BEST]]Thread %d grabbing data %f to replace %f from (%d, %d)\n", threadIdx.x, sharedFitnessBest[threadIdx.x + j], sharedFitnessBest[threadIdx.x],
																						//threadIdx.x + j, threadIdx.x);
					sharedFitnessBest[threadIdx.x] = sharedFitnessBest[threadIdx.x + j];
					sharedTempIndicesBest[threadIdx.x] = sharedTempIndicesBest[threadIdx.x + j];
				}

				if (sharedFitnessWorst[threadIdx.x] == -1 ||
					(sharedFitnessWorst[threadIdx.x] < sharedFitnessWorst[threadIdx.x + j] && sharedFitnessWorst[threadIdx.x + j] != -1))
				{				
					//printf("\t[[WORST]]Thread %d grabbing data %f to replace %f from (%d, %d)\n", threadIdx.x, sharedFitnessWorst[threadIdx.x + j], sharedFitnessWorst[threadIdx.x],
																						//threadIdx.x + j, threadIdx.x);
					sharedFitnessWorst[threadIdx.x] = sharedFitnessWorst[threadIdx.x + j];
					sharedTempIndicesWorst[threadIdx.x] = sharedTempIndicesWorst[threadIdx.x + j];
				}
			}
			
			__syncthreads();
		}
		
		//Replace the index with -1 in the originals
		if (threadIdx.x == 0)
		{
			sharedIndicesBest[i] = sharedTempIndicesBest[0];
			sharedIndicesWorst[i] = sharedTempIndicesWorst[0];

					//printf("We found the best %d value for swarm %d as %f at index %d\n", i, blockIdx.x, sharedFitnessBest[0], __float2int_rn(sharedIndicesBest[i]));
		//printf("We found the worst %d value for swarm %d as %f at index %f\n", i, blockIdx.x, sharedFitnessWorst[0], __float2int_rn(sharedIndicesWorst[i]));

			bestSwapIndices[blockIdx.x * numToSwap + i] = __float2int_rn(sharedIndicesBest[i]);
			worstSwapIndices[blockIdx.x * numToSwap + i] = __float2int_rn(sharedIndicesWorst[i]);

			//printf("Wrote out best value as %d to index %d\n", bestSwapIndices[blockDim.x * numToSwap + i], blockIdx.x * numToSwap + i);
		}

		if (threadIdx.x == 0)
		{
			sharedFitnessOriginal[__float2int_rz(sharedIndicesBest[i])] = -1.0f;
			sharedFitnessOriginal[__float2int_rz(sharedIndicesWorst[i])] = -1.0f;
		}

		__syncthreads();

		sharedFitnessBest[threadIdx.x] = sharedFitnessOriginal[threadIdx.x];
		sharedFitnessWorst[threadIdx.x] = sharedFitnessOriginal[threadIdx.x];

		sharedTempIndicesBest[threadIdx.x] = threadIdx.x;
		sharedTempIndicesWorst[threadIdx.x] = threadIdx.x;

		__syncthreads();

	}//for...

}

/* ClampVelocity
 *
 * Clamps a particle's velocity along one dimension to the
 * prescribed level (currently: half of the number of machines)
 */
__device__ float ClampVelocity(int numMachines, float velocity)
{
	float clamp = 0.4f * numMachines;

	if (velocity > clamp)
		velocity = clamp;
	else if (velocity < -clamp)
		velocity = -clamp;

	return velocity;
}

/* ClampPosition
 *
 * Clamps a particle's position along one dimension ensuring
 * it cannot go outside the valid bounds of the solution space: [0, numMachines - 1].
 */
__device__ float ClampPosition(int numMachines, float position)
{
	if (position < 0.0f)
		position = 0.0f;
	else if (position > numMachines - 1)
		position = (float) numMachines - 1;

	return position;
}

__device__ float Sign(float val)
{
	return val < 0.0f ? -1.0f : val > 0.0f ? 1.0f : 0.0f;
}

__device__ float OmegaFunc(float pos, float fGBest, int numMachines)
{
	float val;
	float dis;

	dis = pos - fGBest;

	val = Sign(dis) * (1 - fabs(dis / numMachines));

	return val;
}

__device__ float FindRepulsiveFactor(float fGBest, float pos, float gBest, int numMachines)
{
	if (pos < fGBest < gBest || gBest < fGBest < pos)
		return -OmegaFunc(pos, fGBest, numMachines);
	else
		return OmegaFunc(pos, fGBest, numMachines);
}

__global__ void UpdateVelocityAndPosition(int numSwarms, int numParticles, int numMachines, int numTasks, float *velocity, float *position, 
										  float *pBestPosition, float *gBestPosition, float *rands, ArgStruct args)
{
	int threadID = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float newVel;
	float currPos;
	//int randOffset;
	int totalParticles = numSwarms * numParticles;
	int gBestOffset;
	int mySwarm;
	int gNeighborOffset;

	//Each thread is responsible for updating one dimension of one particle's 
	if (threadID < __mul24(totalParticles, numTasks))
	{
		//First, figure out what swarm we are covering and who our neighbor is...
		mySwarm = threadID / (numParticles * numTasks);

		gNeighborOffset = mySwarm < numSwarms - 1 ? mySwarm + 1 : 0;


		//The last we are covering is the threadID % numTasks
		//myTask = (threadID / numParticles) % numTasks;

		//Two separate random numbers for every dimension for each particle each iteration.
		//randOffset = ;

		//The swarm this particle belongs to simply the number of threads handling each swarm (numParticles * numTasks)
		//divided by this thread's threadID.
		gBestOffset = (mySwarm * numTasks) + (threadID / numParticles) % numTasks;
		gNeighborOffset = (gNeighborOffset * numTasks) + (threadID / numParticles) % numTasks;
		//gBestOffset += myTask;

		currPos = position[threadID];
		newVel = velocity[threadID];

		newVel *= args.x;		
		newVel += args.z * rands[threadID * 2] * (pBestPosition[threadID] - currPos);
		newVel += args.w * rands[(threadID * 2) + 1] * (gBestPosition[gBestOffset] - currPos);	

		if (mySwarm % 2 == 0)
			newVel += 0.5f * rands[(threadID * 2) + 1] * FindRepulsiveFactor(gBestPosition[gNeighborOffset], gBestPosition[gBestOffset], position[threadID], numMachines);

		//Write out our velocity
		newVel = ClampVelocity(numMachines, newVel);
		velocity[threadID] = newVel;

		//Update the position
		currPos += newVel;
		currPos = ClampPosition(numMachines, currPos);
		position[threadID] = currPos;
	}
}

void InitTexture()
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	texETCMatrix.normalized = false;
	texETCMatrix.filterMode = cudaFilterModePoint;
	texETCMatrix.addressMode[0] = cudaAddressModeClamp;
    texETCMatrix.addressMode[1] = cudaAddressModeClamp;

	cudaMallocArray(&cuArray, &channelDesc, GetNumMachines(), GetNumTasks());
	cudaMemcpyToArray(cuArray, 0, 0, hETCMatrix, sizeof(float) * GetNumMachines() * GetNumTasks(), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texETCMatrix, cuArray, channelDesc);
}

void ClearTexture()
{	
	cudaFreeArray(cuArray);
	cudaUnbindTexture(texETCMatrix);
}

/* GenerateRandomNumbers
 *
 * Generates as many iterations worth of random numbers such that the overall size
 * of random numbers generated does not exceed maxGen.
 *
 * Returns the number of iterations covered by this generation of random numbers.
 */
int GenerateRandomNumbers(int totalParticles, int numTasks, int iterationsRemaining, float *dRand)
{
	int totalRandsPerIter;
	int numIterationsCovered;

	totalRandsPerIter = totalParticles * numTasks * 2;

	numIterationsCovered = MAX_RAND_GEN / (totalRandsPerIter * sizeof(float));
	numIterationsCovered = numIterationsCovered == 0 ? 1 : numIterationsCovered;
	numIterationsCovered = numIterationsCovered > iterationsRemaining ? iterationsRemaining : numIterationsCovered;

	GenRandsGPU(totalRandsPerIter * numIterationsCovered, dRand);

	return numIterationsCovered;
}

float *MRPSODriver(RunConfiguration *run)
{
	int i, j;
	int threadsPerBlock, threadsPerBlockSwap, numBlocks, numBlocksSwap;
	int totalComponents;
	int numMachines, numTasks;
	ArgStruct args;
	float *gBests = NULL;
	float *gBestsTemp;
	float minVal;
	int swapSharedMem;
	int itersOfRands;
	int itersOfRandsLeft;
	int dRandsOffset;
	int useSharedMemFitness;
	unsigned int free, total;

#ifdef KERNEL_TIMING
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

#ifdef RECORD_VALUES
	gBests = (float *) malloc(run->numIterations * sizeof(float));
	gBestsTemp = (float *) malloc(run->numSwarms * sizeof(float));
#endif

	numMachines = GetNumMachines();
	numTasks = GetNumTasks();
	threadsPerBlock = run->threadsPerBlock;
	totalComponents = run->numSwarms * run->numParticles * numTasks;
	args.x = run->w;
	args.y = run->wDecay;
	args.z = run->c1;
	args.w = run->c2;

	numMachines = GetNumMachines();
	numTasks = GetNumTasks();

	swapSharedMem = (run->numParticles * 5 + run->numParticlesToSwap * 2) * sizeof(float);

	numBlocks = CalcNumBlocks(totalComponents, threadsPerBlock);
	threadsPerBlockSwap = 64;
	numBlocksSwap = CalcNumBlocks(run->numSwarms * run->numParticlesToSwap, threadsPerBlockSwap);

	//cuMemGetInfo(&free, &total);
	//printf("Free: %d, total: %d\n", free, total);

	//Generate the random numbers we need for the initialization...
	//InitRandsGPU();
#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif

	GenRandsGPU(run->numSwarms * run->numParticles * numTasks * 2, dRands);

#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	genRandTime += elapsedTime;
#endif

	//Decide if we're using the shared memory fitness kernel or not
	if (run->numParticles * numMachines * sizeof(float) > 15360)
		useSharedMemFitness = 0;
	else
		useSharedMemFitness = 1;

#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif

	//Initialize our particles.
	InitializeParticles<<<numBlocks, threadsPerBlock>>>(run->numSwarms, run->numParticles, numTasks, numMachines, dGBest, dPBest, dPosition, dVelocity, dRands);

#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	initTime += elapsedTime;
#endif

	itersOfRandsLeft = 0;

	//Run MRPSO GPU for the given number of iterations.
	for (i = 1; i <= run->numIterations; i++)
	{
		//If we need to generate more random numbers then do so now...
		if (!itersOfRandsLeft)
		{
#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif
			itersOfRands = GenerateRandomNumbers(run->numSwarms * run->numParticles, numTasks, run->numIterations - i + 1, dRands);
			itersOfRandsLeft = itersOfRands;

#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	genRandTime += elapsedTime;
#endif
		}

#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif

		//Update the Fitness
		if (useSharedMemFitness)
			UpdateFitnessShared<<<run->numSwarms, run->numParticles, run->numParticles * GetNumMachines() * sizeof(float)>>>(run->numSwarms, run->numParticles, 
																					                                   GetNumTasks(), GetNumMachines(), dPosition, dFitness);
		else
			UpdateFitness<<<run->numSwarms, run->numParticles>>>(run->numSwarms, run->numParticles, numTasks, numMachines, dPosition, dScratch, dFitness);

#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fitnessTime += elapsedTime;
#endif

#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif
		//Update the local and swarm best positions
		UpdateBests<<<run->numSwarms, run->numParticles, run->numParticles * 2 * sizeof(float)>>>(run->numSwarms, run->numParticles, numTasks, dPBest, 
																							      dPBestPosition, dGBest, dGBestPosition,
														                                          dPosition, dFitness);
#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	findBestsTime += elapsedTime;
#endif

#ifdef RECORD_VALUES
		cudaThreadSynchronize();
		cudaMemcpy(gBestsTemp, dGBest, run->numSwarms * sizeof(float), cudaMemcpyDeviceToHost);

		minVal = gBestsTemp[0];

		//Find the minimal gbest value
		for (j = 1; j < run->numSwarms; j++)
		{
			if (gBestsTemp[j] < minVal)
				minVal = gBestsTemp[j];
		}

		gBests[i - 1] = minVal;
#endif

#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif

		//REMINDER: The problem lies in the random number use after a certain number of iterations.
		//Update the Position and Velocity
		dRandsOffset = (itersOfRands - itersOfRandsLeft) * run->numSwarms * run->numParticles * numTasks * 2;
		UpdateVelocityAndPosition<<<numBlocks, threadsPerBlock>>>(run->numSwarms, run->numParticles, numMachines, numTasks, 
																  dVelocity, dPosition, dPBestPosition, dGBestPosition, &dRands[dRandsOffset], args);	
#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	updatePosVelTime += elapsedTime;
#endif

		if (args.x > 0.001f)
			args.x *= run->wDecay;

		if (i % run->iterationsBeforeSwap == 0)
		{

#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif
			//Build up the swap indices for each swarm
			GenerateSwapIndices<<<run->numSwarms, run->numParticles, swapSharedMem>>>(run->numSwarms, run->numParticles, run->numParticlesToSwap, 
			                                                                          dFitness, dBestSwapIndices, dWorstSwapIndices);
#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	determineSwapTime += elapsedTime;
#endif

#ifdef KERNEL_TIMING
	cudaEventRecord(start, 0);
#endif

			//Swap particles between swarms
			SwapBestParticles<<<numBlocksSwap, threadsPerBlockSwap>>>(run->numSwarms, run->numParticles, numTasks, run->numParticlesToSwap, dBestSwapIndices, 
																	  dWorstSwapIndices, dPosition, dVelocity, dPBest, dPBestPosition);
#ifdef KERNEL_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	swapTime += elapsedTime;
#endif
		}

		itersOfRandsLeft--;
	}

	FreeRandsGPU();

#ifdef RECORD_VALUES
	free(gBestsTemp);
#endif

	return gBests;
}


