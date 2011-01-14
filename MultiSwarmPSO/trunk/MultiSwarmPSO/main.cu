#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
#include "gpu_pso.h"
#include "cpu_pso.h"
#include "tests.h"

int main(int argc, char* argv[])
{
	/*
	OpenRunsFile(argv[1]);

	run = GetNextRun();

	RunMRPSO(run);

	data = GetRecordedData();

	best = data[run->numIterations - 1];

	for (i = 1; i < run->numSwarms; i++)
		if (best > data[(run->numIterations * i + 1) - 1])
			best = data[(run->numIterations * i + 1) - 1];

	printf("Best value found: %f\n", best);
	*/

	//TestTex();

	//TestGPUMatch();

	//RunGPURandGenTests();
	
	RunGPUCorrectnessTests();


	getchar();

	return 0;
}
