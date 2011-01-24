#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tests.h"
#include "helper.h"
#include "gpu_pso.h"

void UpdateVelocityPosition(int numSwarms, int numParticles, int numTasks, int numMachines, float *velocity, float *position, 
						    float *pBestPositionVector, float *gBestPositionVector, float w, float c1, float c2)
{
	int i, j, k;
	int particleOffset;
	float newVelocity;	
	
	for (i = 0; i < numSwarms; i++)
	{
		for (j = 0; j < numParticles; j++)
		{
			particleOffset = (i * numParticles * numTasks) + (j * numTasks);

			for (k = 0; k < numTasks; k++)
			{
				newVelocity = w * velocity[particleOffset + k];
				newVelocity += c1 * ( 1.0f * (pBestPositionVector[particleOffset + k] - position[particleOffset + k]) );
				newVelocity += c2 * ( 1.0f * (gBestPositionVector[(i * numTasks) + k] - position[particleOffset + k]) );
	
				if (newVelocity > numMachines * .5f)
					newVelocity = numMachines * .5f;
				else if (newVelocity < numMachines * -0.5f)
					newVelocity = numMachines * -0.5f;

				velocity[particleOffset + k] = newVelocity;
				position[particleOffset + k] += newVelocity;

				if (position[particleOffset + k] < 0)
					position[particleOffset + k] = 0.0f;
				else if (position[particleOffset + k] > numMachines - 1)
					position[particleOffset + k] = (float) numMachines - 1;
			}
		}
	}
}

int TestUpdateVelocityAndPosition()
{
	int passed = 1;
	int i, j;
	float *hVelocity, *dVelocity, *cpuVelocity;
	float *hPosition, *dPosition, *cpuPosition;
	float *hRand, *dRand;
	float *hPBestPosition, *dPBestPosition, *cpuPBestPosition;
	float *hGBestPosition, *dGBestPosition, *cpuGBestPosition;
	int numParticles, numSwarms, numTasks, numMachines;
	int threadsPerBlock, numBlocks;
	float w, c1, c2;
	ArgStruct arg;

	numSwarms = 20;
	numParticles = 64;
	numTasks = 200;
	numMachines = 8;
	arg.x = w = 1.0f;
	arg.z = c1 = 2.0f;
	arg.w = c2 = 1.4f;

	threadsPerBlock = 256;
	numBlocks = CalcNumBlocks(numSwarms * numParticles * numTasks, threadsPerBlock);

	printf("\tRunning GPU position and velocity update test...\n");

	srand((unsigned int) time(NULL));

	hPBestPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	cpuPBestPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	hGBestPosition = (float *) malloc(numSwarms * numTasks * sizeof(float));
	cpuGBestPosition = (float *) malloc(numSwarms * numTasks * sizeof(float));

	hVelocity = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	cpuVelocity= (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	hPosition= (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));
	cpuPosition = (float *) malloc(numSwarms * numParticles * numTasks * sizeof(float));

	hRand = (float *) malloc(numSwarms * numParticles * numTasks * 2 * sizeof(float));

	cudaMalloc((void **) &dPBestPosition, numSwarms * numParticles * numTasks * sizeof(float));
	cudaMalloc((void **) &dGBestPosition, numSwarms * numTasks * sizeof(float));
	cudaMalloc((void **) &dVelocity, numSwarms * numParticles * numTasks * sizeof(float));
	cudaMalloc((void **) &dPosition, numSwarms * numParticles * numTasks * sizeof(float));
	cudaMalloc((void **) &dRand, numSwarms * numParticles * numTasks * 2 * sizeof(float));

	//Generate our random numbers
	for (i = 0; i < numSwarms * numParticles * numTasks * 2; i++)
		hRand[i] = 1.0f;

	//Randomly generate our PBest values and positions, velocities, and positions
	for (i = 0; i < numSwarms * numParticles; i++)
	{
		for (j = 0; j < numTasks; j++)
		{
			hPBestPosition[i * numTasks + j] = (float) (rand() % numMachines);
			cpuPBestPosition[i * numTasks + j] = hPBestPosition[i * numTasks + j];

			hPosition[i * numTasks + j] = (float) (rand() % numMachines);
			hVelocity[i * numTasks + j] = (float) (rand() % numMachines);

			cpuPosition[i * numTasks + j] = hPosition[i * numTasks + j];
			cpuVelocity[i * numTasks + j] = hVelocity[i * numTasks + j];
		}
	}

	//Randomly generate our GBest values and positions.
	for (i = 0; i < numSwarms; i++)
	{
		for (j = 0; j < numTasks; j++)
		{
			hGBestPosition[i * numTasks + j] = (float) (rand() % numMachines);
			cpuGBestPosition[i * numTasks + j] = hGBestPosition[i * numTasks + j];
		}
	}	

	//Dump the data to the GPU
	cudaMemcpy(dPBestPosition, hPBestPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGBestPosition, hGBestPosition, numSwarms * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dPosition, hPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocity, hVelocity, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dRand, hRand, numSwarms * numParticles * numTasks * 2 * sizeof(float), cudaMemcpyHostToDevice);

	//Compute the reference sequential solution.
	UpdateVelocityPosition(numSwarms, numParticles, numTasks, numMachines, cpuVelocity, cpuPosition, 
						    cpuPBestPosition, cpuGBestPosition, w, c1, c2);

	//Compute the GPU solution
	UpdateVelocityAndPosition<<<numBlocks, threadsPerBlock>>>(numSwarms, numParticles, numMachines, numTasks, dVelocity, dPosition, 
																	 dPBestPosition, dGBestPosition, dRand, arg);
	cudaThreadSynchronize();

	cudaMemcpy(hPosition, dPosition, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hVelocity, dVelocity, numSwarms * numParticles * numTasks * sizeof(float), cudaMemcpyDeviceToHost);

	//Ensure the position and velocities values match;
	for (i = 0; i < numSwarms * numParticles; i++)
	{
		for (j = 0; j < numTasks; j++)
		{
			if(abs(hVelocity[i * numTasks + j] - cpuVelocity[i * numTasks + j]) > ACCEPTED_DELTA)
			{
				printf("\t[ERROR] - GPU Velocity value for particle %d, element %d was: %f (expected: %f)\n", i, j, 
					   hVelocity[i * numTasks + j], cpuVelocity[i * numTasks + j]);
				passed = 0;
			}

			if(abs(hPosition[i * numTasks + j] - cpuPosition[i * numTasks + j]) > ACCEPTED_DELTA)
			{
				printf("\t[ERROR] - GPU Position value for particle %d, element %d was: %f (expected: %f)\n", i, j, 
					   hPosition[i * numTasks + j], cpuPosition[i * numTasks + j]);
				passed = 0;
			}
		}
	}

	PrintTestResults(passed);

	free(hVelocity);
	free(cpuVelocity);
	free(hPosition);
	free(cpuPosition);
	free(hPBestPosition);
	free(cpuPBestPosition);
	free(hGBestPosition);
	free(cpuGBestPosition);
	free(hRand);
	cudaFree(dVelocity);
	cudaFree(dPosition);
	cudaFree(dPBestPosition);
	cudaFree(dGBestPosition);
	cudaFree(dRand);

	return passed;
}

void RunPositionVelocityTests()
{
	int passed = 1;

	printf("\nStarting GPU position and velocity tests...\n\n");

	passed &= TestUpdateVelocityAndPosition();

	if (passed)
		printf("[PASSED] All position and velocity tests passed!\n\n");
	else
		printf("[FAILED] Position and velocity tests failed!\n\n");
}