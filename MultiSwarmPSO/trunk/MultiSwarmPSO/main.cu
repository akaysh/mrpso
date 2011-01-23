#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
#include "gpu_pso.h"
#include "cpu_pso.h"
#include "tests.h"

int main(int argc, char* argv[])
{
	//TestGPUMatch();

	//RunGPURandGenTests();

	//printf((cudaGetErrorString(cudaGetLastError())));
	//InitCUDA();
	
	//RunGPUCorrectnessTests();

	InitCUDA();

	TestMakespan("runs.txt");

	

	//getchar();

	printf((cudaGetErrorString(cudaGetLastError())));

	

	return 0;
}
