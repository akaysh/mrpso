#include <stdio.h>
#include "tests.h"
#include "helper.h"

void RunGPUCorrectnessTests()
{
	printf("-----------------------------------\n");
	printf("Starting all GPU Correctness Tests.\n");
	printf("-----------------------------------\n\n");

	RunGPUTextureTests();
	RunSwarmFunctionTests();

}