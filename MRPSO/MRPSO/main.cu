#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
#include "cpu_pso.h"
#include "tests.h"

int main(int argc, char* argv[])
{
	int i;
	float best;
	RunConfiguration *run;
	float *data;

	OpenRunsFile(argv[1]);

	run = GetNextRun();

	RunMRPSO(run);

	data = GetRecordedData();

	best = data[run->numIterations - 1];

	for (i = 1; i < run->numSwarms; i++)
		if (best > data[(run->numIterations * i + 1) - 1])
			best = data[(run->numIterations * i + 1) - 1];

	printf("Best value found: %f\n", best);

	getchar();

	return 0;
}