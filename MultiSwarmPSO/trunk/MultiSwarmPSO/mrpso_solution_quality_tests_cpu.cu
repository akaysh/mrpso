#include <stdio.h>
#include "tests.h"
#include "helper.h"
#include "cpu_pso.h"

void TestMRPSOCPUResults(char *filename)
{
	int opened;
	FILE *outFile;
	float *data;
	RunConfiguration *run;

	opened = OpenRunsFile(filename);

	if (opened)
	{
		outFile = fopen("makespan_results.csv", "w");

		printf("Starting Makespan PSO Performance Tests...\n");

		if (outFile != NULL)
		{
			fprintf(outFile, "task_file,machine_file,iteration,gbest,energy\n");
			run = GetNextRun();

			while (run->numSwarms >= 0)
			{
				RunMRPSO(run);

				//data = GetRecordedData();

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
