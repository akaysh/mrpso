#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
#include "gpu_pso.h"
#include "cpu_pso.h"
#include "tests.h"

int main(int argc, char* argv[])
{	
	//RunGPUCorrectnessTests();

	InitCUDA();

	TestMakespan("runs.txt");

	//TestSolutionQuality("runs.txt");

	return 0;
}

