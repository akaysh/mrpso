#ifndef _GPU_PSO_H_
#define _GPU_PSO_H_

#include <cuda_runtime.h>

float *MRPSODriver(RunConfiguration *run);
void InitTexture();
void ClearTexture();

__global__ void InitializeParticles(int numSwarms, int numParticles, int numTasks, int numMachines, float *gBests, float *pBests, float *position, 
									float *velocity, float *randNums);

__global__ void SwapBestParticles(int numSwarms, int numParticles, int numTasks, int numToSwap, int *bestSwapIndices, int *worstSwapIndices, 
								  float *position, float *velocity, float *pBest, float *pBestPosition);

__global__ void UpdateVelocityAndPosition(int numSwarms, int numParticles, int numMachines, int numTasks, float *velocity, float *position, 
										  float *pBestPosition, float *gBestPosition, float *rands, ArgStruct args);

__global__ void UpdateBests(int numSwarms, int numParticles, int numTasks, float *pBest, float *pBestPositions, float *gBest, float *gBestPositions,
							float *position, float *fitness);

__global__ void GenerateSwapIndices(int numSwarms, int numParticles, int numToSwap, float *fitness, int *bestSwapIndices, int *worstSwapIndices);

__global__ void UpdateFitness(int numSwarms, int numParticles, int numTasks, int numMachines, float *position, float *scratch, float *fitness);

#endif