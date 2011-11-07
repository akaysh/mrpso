#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cuda.h>
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

	fprintf(timeOutFile, "num_tasks,num_machines,num_swarms,particles_per_swarm,num_iterations,init_time,update_pos_vel_time,fitness_time,update_bests_time,determine_swaps_time,swap_time,rand_time,cpu_time,gpu_time\n");

	while (run->numSwarms != -1)
	{
		gpuTime = 0.0f;
		cpuTime = 0.0f;
		ResetTimers();

		printf("Running test with %d tasks and %d machines with %d swarms and %d particles...\n", run->taskFile, run->machineFile, run->numSwarms, run->numParticles);

		for (i = 0; i < run->numTests; i++)
		{						
			cudaThreadSynchronize();
			
			cutResetTimer(timer);
			cutStartTimer(timer);

			InitRandsGPU();
			AllocateGPUMemory(run);
			InitTexture();
			
			MRPSODriver(run);

			cudaThreadSynchronize();
			cutStopTimer(timer);	

			FreeGPUMemory();
			ClearTexture();

			gpuTime += cutGetTimerValue(timer);

			cutResetTimer(timer);
			cutStartTimer(timer);

			RunMRPSO(run);

			cutStopTimer(timer);

			cpuTime += cutGetTimerValue(timer);

			printf(".");
			fflush(stdout);
		}

		FreeCPUMemory();

		printf("Completed run!\n\n");

		gpuTime /= run->numTests;
		cpuTime /= run->numTests;
		initTime /= run->numTests;
		swapTime /= run->numTests;
		determineSwapTime /= run->numTests; 
		findBestsTime /= run->numTests;
		updatePosVelTime /= run->numTests;
		fitnessTime /= run->numTests;
		genRandTime /= run->numTests;

		fprintf(timeOutFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", run->taskFile, run->machineFile, run->numSwarms, run->numParticles, run->numIterations, 
			    initTime, updatePosVelTime, fitnessTime, findBestsTime, determineSwapTime, swapTime, genRandTime, cpuTime, gpuTime);

		run = GetNextRun();
	}

	fclose(timeOutFile);
}

void TestSolutionQuality(char *filename)
{
	int i, j;
	RunConfiguration *run;
	FILE *qualFile;

	float *mrpsoRet, *psoRet, fcfsRet;

	float mrpsoComp, psoComp;

	qualFile = fopen("qual_results.csv", "w");

	OpenRunsFile(filename);
	fprintf(qualFile, "numTasks,numMachines,numSwarms,numParticles,numIterations,MRPSO,PSO,FCFS\n");

	run = GetNextRun();

	while (run->numSwarms != -1)
	{
		mrpsoComp = 0.0f;
		psoComp = 0.0f;

		for (i = 0; i < 10; i++)
		{
			FreeCPUMemory();
			GenData(run);

			//Run the same test numTests times
			for (j = 0; j < run->numTests; j++)
			{
				InitRandsGPU();
				AllocateGPUMemory(run);
				InitTexture();
			
				mrpsoRet = MRPSODriver(run);

				FreeGPUMemory();
				ClearTexture();

				psoRet = RunMakespanPSO(run->numParticles, GetNumTasks(), GetNumMachines(), run->w, run->wDecay, run->c1, run->c2, run->numIterations, BASIC);

				fcfsRet = GetFCFSMapping(GetNumTasks(),GetNumMachines());

				mrpsoComp += mrpsoRet[run->numIterations - 1] / fcfsRet;
				psoComp += psoRet[(run->numIterations - 1) * 2] / fcfsRet;

				free(mrpsoRet);
				free(psoRet);
			}
		}

		//printf("%f versus %f\n", mrpsoComp/(run->numTests * 10), psoComp/(run->numTests * 10));
		printf("%f\n", psoComp/(run->numTests * 10));

		run = GetNextRun();

	}
	fclose(qualFile);
}
