#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
#include "gpu_pso.h"
#include "cpu_pso.h"

void TestMakespan(char *filename)
{
	int i;
	RunConfiguration *run;
	FILE *outFile;

	OpenRunsFile(filename);

	run = GetNextRun();

	while (run != NULL)
	{
		for (i = 0; i < run->numTests; i++)
		{
			GenerateRandsGPU(initializationRandCount + iterationRandCount, dRands);
			MRPSODriver(run);
		}


	}

	



}