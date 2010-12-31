#include <cutil.h>
#include <cuda_runtime.h>

__device__ float ComputeMakespan(float *matching)
{


}

__global__ void SwapBestParticles(int numSwarms, int numParticles, int *swapIndices, float *position, float *velocity, float *fitness, float *newPosition, float *newVelocity, float *newFitness)
{
	//Perform the swap into the new_____ arrays (double buffering)


}

__global__ void RunIteration(int numSwarms, int numParticles, float *ETCMatrix, float *position, float *velocity, float *fitness, float *pBest, float *pBestPosition, float *gBest, float *gBestPosition)
{



}

float *MRPSODriver()
{
	float *matching = NULL;

	return matching;
}