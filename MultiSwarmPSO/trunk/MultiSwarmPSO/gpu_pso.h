#ifndef _GPU_PSO_H_
#define _GPU_PSO_H_

#include <cuda_runtime.h>

__global__ void SwapBestParticles(int numSwarms, int numParticles, int numTasks, int numToSwap, int *bestSwapIndices, int *worstSwapIndices, float *position, float *velocity);

#endif