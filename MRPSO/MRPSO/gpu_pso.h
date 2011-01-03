#ifndef _GPU_PSO_H_
#define _GPU_PSO_H_

#include <cuda_runtime.h>

void TestTex();
void TestGPUMatch();

__device__ float CalcMakespan(int numTasks, int numMachines, float *matching, float *scratch);

#endif