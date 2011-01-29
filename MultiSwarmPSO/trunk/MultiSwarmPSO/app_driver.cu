#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include "helper.h"
#include "gpu_pso.h"
#include "cpu_pso.h"

void TestMakespan(char *filename)
{
	int i;
	RunConfiguration *run;
	FILE *outFile;
	float totalTime;
	unsigned int timer;

	OpenRunsFile(filename);

	run = GetNextRun();

	cutCreateTimer(&timer);

	while (run->numSwarms != -1)
	{
		totalTime = 0.0f;
		ResetTimers();

		for (i = 0; i < run->numTests; i++)
		{			
			BuildMachineList(run->machineFile);
			BuildTaskList(run->taskFile);
			GenerateETCMatrix();	
			
			cutResetTimer(timer);
			cutStartTimer(timer);

			InitRandsGPU();
			AllocateGPUMemory(run);
			InitTexture();

			MRPSODriver(run);

			cutStopTimer(timer);	

			FreeCPUMemory();
			FreeGPUMemory();
			ClearTexture();

			totalTime += cutGetTimerValue(timer);
		}


		printf("Total time: %.3f ms\n", totalTime / run->numTests);
		run = GetNextRun();


	}

	



}