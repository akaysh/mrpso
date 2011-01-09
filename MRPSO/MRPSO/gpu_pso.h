#ifndef _GPU_PSO_H_
#define _GPU_PSO_H_

#include <cuda_runtime.h>

void TestTex();
void TestGPUMatch();

__global__ void SwapBestParticles(int numSwarms, int numTasks, int numToSwap, int *bestSwapIndices, int *worstSwapIndices, float *position, float *velocity);

#endif