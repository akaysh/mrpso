#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "curand.h"

__global__ void SwapBestParticles(int numSwarms, int numTasks, int numToSwap, int *bestSwapIndices, int *worstSwapIndices, float *position, float *velocity)
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
		bestIndex = __mul24(bestSwapIndices[mySwarmIndex + myIndex], numTasks);
		worstIndex = __mul24(worstSwapIndices[neighborSwarmIndex + myIndex], numTasks);

		printf("Thread %i swapping %i with %i\n", threadID, bestIndex, worstIndex);
		
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