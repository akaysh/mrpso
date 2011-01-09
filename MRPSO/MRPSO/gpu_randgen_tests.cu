#include <stdio.h>
#include "tests.h"
#include "helper.h"

void RunGPURandGenTests()
{
	int i;
	RunConfiguration *run;
	float *hRands;

	OpenRunsFile("runs.txt");
	run = GetNextRun();
	GenerateRandsGPU(run);

	hRands = (float *) malloc(run->numParticles * run->numSwarms * run->numIterations * 2 * sizeof(float));

	cudaMemcpy(hRands, dRands, run->numParticles * run->numSwarms * run->numIterations * 2 * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < 10; i++)
		printf("%f ", hRands[i]);

	FreeCPUMemory();
	FreeGPUMemory();
	free(hRands);

}