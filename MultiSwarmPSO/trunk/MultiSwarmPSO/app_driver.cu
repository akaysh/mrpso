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
	FILE *timeOutFile;
	float gpuTime, cpuTime;
	unsigned int timer;

	timeOutFile = fopen("perf_results.csv", "w");

	OpenRunsFile(filename);

	run = GetNextRun();

	cutCreateTimer(&timer);

	fprintf(timeOutFile, "num_swarms,particles_per_swarm,num_iterations,init_time,update_pos_vel_time,fitness_time,update_bests_time,determine_swaps_time,swap_time,rand_time,cpu_time,gpu_time\n");

	while (run->numSwarms != -1)
	{
		gpuTime = 0.0f;
		cpuTime = 0.0f;
		ResetTimers();

		for (i = 0; i < run->numTests; i++)
		{			
			BuildMachineList(run->machineFile);
			BuildTaskList(run->taskFile);
			GenerateETCMatrix();	
			
			cudaThreadSynchronize();
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

			gpuTime += cutGetTimerValue(timer);

			BuildMachineList(run->machineFile);
			BuildTaskList(run->taskFile);
			GenerateETCMatrix();

			cutResetTimer(timer);
			cutStartTimer(timer);

			//RunMRPSO(run);

			cutStopTimer(timer);

			cpuTime += cutGetTimerValue(timer);

			FreeCPUMemory();
		}

		gpuTime /= run->numTests;
		cpuTime /= run->numTests;
		swapTime /= run->numTests;
		determineSwapTime /= run->numTests; 
		findBestsTime /= run->numTests;
		updatePosVelTime /= run->numTests;
		fitnessTime /= run->numTests;

		fprintf(timeOutFile, "%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", run->numSwarms, run->numParticles, run->numIterations, initTime, updatePosVelTime, fitnessTime, findBestsTime,
			    determineSwapTime, swapTime, genRandTime, cpuTime, gpuTime);

		run = GetNextRun();
	}

	fclose(timeOutFile);
}
