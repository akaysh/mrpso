#include <stdio.h>
#include <stdlib.h>
#include "helper.h"

/* Implements a First-Come-First-Serve task mapping system
 * that uses the minimum-completion-time rating to match
 * tasks with machines in a grid environment. 
 */
float GetFCFSMapping(int numTasks, int numMachines)
{
	int i, j;
	float *matching;
	float *MAT;
	float curr, currBestTime;
	int currBestMachine;

	matching = (float *) malloc(numTasks * sizeof(float));
	MAT = (float *) calloc(numMachines, sizeof(float));

	//Find the current best machine based on MCT for each task...
	for (i = 0; i < numTasks; i++)
	{
		currBestTime = MAT[0] + hETCMatrix[i * numMachines];
		currBestMachine = 0;

		for (j = 1; j < numMachines; j++)
		{
			curr = MAT[j] + hETCMatrix[i * numMachines + j];
			if (curr < currBestTime)
			{
				currBestTime = curr;
				currBestMachine = j;
			}
		}

		//Assign this task to the given machine and increase the MAT
		matching[i] = currBestMachine;
		MAT[currBestMachine] += hETCMatrix[i * numMachines + currBestMachine];
	}

	free(MAT);

	return ComputeMakespan(matching, numTasks);
}

