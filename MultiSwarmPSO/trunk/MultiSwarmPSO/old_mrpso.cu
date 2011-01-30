#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "curand.h"

texture<float, 2, cudaReadModeElementType> texETCMatrix;

extern __shared__ float sharedGBestPosition[];

__device__ float ClampVelocityOld(int numMachines, float velocity)
{
	float clamp = 0.5f * numMachines;

	if (velocity > clamp)
		velocity = clamp;
	else if (velocity < -clamp)
		velocity = -clamp;

	return velocity;
}

__device__ float ClampPositionOld(int numMachines, float position)
{
	if (position < 0.0f)
		position = 0.0f;
	else if (position > numMachines - 1)
		position = (float) numMachines - 1;

	return position;
}

__device__ void UpdateVelocityAndPositionOld(int numSwarms, int numParticles, int numMachines, int numTasks, float *velocity, float *position, float *pBestPosition, 
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
		newVel = ClampVelocityOld(numMachines, newVel);

		//Write out our velocity to global memory.
		velocity[swarmOffset + (i * numParticles) + threadIdx.x] = newVel;

		//Might as well update the position along this dimension while we're at it.
		currPos += newVel;
		currPos = ClampPositionOld(numMachines, currPos);
		position[swarmOffset + (i * numParticles) + threadIdx.x] = currPos;
	}
}

__device__ float CalcMakespanOld(int numTasks, int numMachines, float *matching, float *scratch)
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
