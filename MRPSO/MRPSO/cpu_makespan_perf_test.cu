/* Contains all the tests for measuring
 * the performance of the Makespan, Energy,
 * and Hybrid (VE) PSO implementations
 */

#include "cpu_pso.h"
#include "tests.h"
#include "helper.h"

#define NUM_TESTS 20

void TestMakespan(FILE *outFile, RunConfiguration run)
{
	int i, j;
	float *iterationGBestC, *currIterationC = NULL;

	iterationGBestC = (float *) calloc(run.numIterations * 2, sizeof(float));

	if (hMachines != NULL && hTasks != NULL)
	{
		for (i = 0; i < NUM_TESTS; i++)
		{
			currIterationC = RunMakespanPSO(run.numParticles, GetNumTasks(), GetNumMachines(), run.w, run.wDecay, run.c1, run.c2, run.numIterations, BASIC);

			//Add the current iteration values to the running totals.
			for (j = 0; j < run.numIterations; j++)
			{
				iterationGBestC[j * 2] += currIterationC[j * 2];
				iterationGBestC[j * 2 + 1] += currIterationC[j * 2 + 1];
			}

			printf(".");
			fflush(stdout);

			free(currIterationC);
		}

		printf("\n");

		//Average the iteration results and write it all out to the output file
		for (i = 0; i < run.numIterations; i++)
		{
			iterationGBestC[i * 2] /= (float) NUM_TESTS;
			iterationGBestC[i * 2 + 1] /= (float) NUM_TESTS;
			fprintf(outFile, "%s,%s,%d,%.8f,%.8f\n", run.taskFile, run.machineFile, i, iterationGBestC[i * 2], iterationGBestC[i * 2 + 1]);
		}

		free(iterationGBestC);
	}
}


void TestMakespanResults(char *filename)
{
	int i;
	int opened;
	FILE *outFile;
	RunConfiguration run;

	opened = OpenRunsFile(filename);

	if (opened)
	{
		outFile = fopen("makespan_results.csv", "w");

		printf("Starting Makespan PSO Performance Tests...\n");

		if (outFile != NULL)
		{
			fprintf(outFile, "task_file,machine_file,iteration,gbest,energy\n");
			run = GetNextRun();

			while (run.numSwarms >= 0)
			{
				TestMakespan(outFile, run);

				run = GetNextRun();
			}

			CloseRunsFile();

			fclose(outFile);
		}
		else
			printf("[Error] Could not open file makespan_results.csv for writing.\n");
	}
	else
		printf("[Error] Could not open %s for reading.\n", filename);
}


