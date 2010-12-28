#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "helper.h"

int numMachines, numTasks;

float *hETCMatrix = NULL;
Machine *hMachines = NULL;
Task *hTasks = NULL;

int GetNumMachines()
{
	return numMachines;
}

int GetNumTasks()
{
	return numTasks;
}

/* PrintMachines
 *
 * Prints out the list of Machines and their statistics to the screen.
 */
void PrintMachines()
{
	int i;

	printf("Displaying %d machine", numMachines);

	if (numMachines > 1)
		printf("s");

	printf("...\n\n");

	printf("ID\tMIPS\t\tEnergy Use\n");
	printf("--\t----\t\t----------\n");

	for (i = 0; i < numMachines; i++)
			printf("%d\t%.2lf\t\t%.2lf\n", i, hMachines[i].MIPS, hMachines[i].energy); 

	printf("\n");
}

/* PrintTasks
 *
 * Prints out the list of Tasks and their characteristics to the screen.
 */
void PrintTasks()
{
	int i;

	printf("Displaying %d task", numTasks);

	if (numTasks > 1)
		printf("s");

	printf("...\n\n");

	printf("ID\tLength\n");
	printf("--\t------\n");

	for (i = 0; i < numTasks; i++)
			printf("%d\t%.2lf\n", i, hTasks[i].length); 

	printf("\n");
}

/* PrintETCMatrix
 *
 * Prints out the ETC Matrix for the current set of hMachines and hTasks
 * to the screen.
 */
void PrintETCMatrix()
{
	int i, j;

	if (hETCMatrix != NULL)
	{
		printf("Displaying ETC Matrix...\n\n");
		printf("\t");

		for (i = 0; i < numMachines; i++)
			printf("M%d\t", i);

		printf("\n");
			
		for (i = 0; i < numTasks; i++)
		{
			printf("%d\t", i);

			for (j = 0; j < numMachines; j++)
				printf("%1.3lf\t", hETCMatrix[(i * numMachines) + j]);

			printf("\n");
		}
	}
	else
		printf("[Error] ETC Matrix has not been initialized.\n");
}

/* BuildMachineList
 *
 * Reads machine definitions from the input file and 
 * forms the list of Machines in the system.
 */
Machine* BuildMachineList(char *filename)
{
	FILE *file;
	int currMachine;
	float MIPS, energy;
	char line[512], *ptr;

	file = fopen(filename, "r");

	if (file != NULL)
	{
		//First integer tells us how many hMachines there are.
		fgets(line, sizeof line, file);
		numMachines = atoi(line);

		if (hMachines != NULL)
			free(hMachines);

		hMachines = (Machine *) malloc(numMachines * sizeof(Machine));
		currMachine = 0;

		//Each subsequent line contains the machine definitions in the form:
		//MIPS EnergyUse
		while(fgets(line, sizeof line, file) != NULL)
		{
			MIPS = (float) strtod(line, &ptr);
			energy = (float) strtod(ptr, &ptr);

			hMachines[currMachine].MIPS = MIPS;
			hMachines[currMachine++].energy = energy;
		}

		if (currMachine != numMachines)
		{
			printf("\tWarning: Number of hMachines in file less than expected.\n");
			numMachines = currMachine;
		}		

		fclose(file);
	}
	else
	{
		printf("Could not open file %s\n", filename);
		hMachines = NULL;
	}

	return hMachines;
}

/* BuildTaskList
 *
 * Reads tast definitions from the input file and 
 * forms the list of hTasks requiring mapping.
 */
Task* BuildTaskList(char *filename)
{
	FILE *file;
	int currTask;
	float length;
	char line[512], *ptr;

	file = fopen(filename, "r");

	if (file != NULL)
	{
		//First integer tells use how many hTasks there are.
		fgets(line, sizeof line, file);
		numTasks = atoi(line);

		if (hTasks != NULL)
			free(hTasks);

		hTasks = (Task *) malloc(numTasks * sizeof(Task));
		currTask = 0;

		while (fgets(line, sizeof line, file) != NULL)
		{
			length = (float) strtod(line, &ptr);
			hTasks[currTask++].length = length;
		}

		if (currTask != numTasks)
		{
			printf("\n[Warning] Number of hTasks in file less than expected.\n");
			numTasks = currTask;
		}

		fclose(file);
	}
	else
	{
		printf("Could not open file %s\n", filename);
		hTasks = NULL;
	}

	return hTasks;
}

/* FreeMemory
 *
 * Frees the memory allocated by building the Machine and Task
 * lists as well as the ETC Matrix.
 */
void FreeMemory()
{
	if (hMachines != NULL)
		free(hMachines);

	if (hTasks != NULL)
		free(hTasks);

	if (hETCMatrix != NULL)
		free(hETCMatrix);

	hMachines = NULL;
	hTasks = NULL;
	hETCMatrix = NULL;
}

/* GenerateETCMatrix
 *
 * Generates the ETC Matrix covering the given hTasks and hMachines.
 * The value at (i, j) provides the time taken to compute task i
 * on machine j. This value is arrived at simply by taking the
 * length of task i and dividing it by the MIPS rating of machine j.
 */
float* GenerateETCMatrix()
{
	int i, j;

	hETCMatrix = (float *) malloc(numTasks * numMachines * sizeof(float));

	for (i = 0; i < numTasks; i++)
	{
		for (j = 0; j < numMachines; j++)
		{
			hETCMatrix[(i * numMachines) + j] = hTasks[i].length / hMachines[j].MIPS;
		}
	}

	return hETCMatrix;
}

/* GenDiscreteCoord
 *
 * Generates a discrete coordinate from a continuous coordinate.
 */
int DiscreteCoord(float value)
{
	return (int) floor(value + 0.5);
}

/* ComputeMakespan
 *
 * Computes the makespan of the given task - machine matching.
 */
float ComputeMakespan(float *matching, int numTasks)
{
	int i;
	float *potentialMakespans;
	float makespan;

	potentialMakespans = (float *) calloc(numTasks, sizeof(float));

	//Each dimension of the Particle's position vector represents a task, the actual
	//position in that dimension represents the machine this task is matched to.
	for (i = 0; i < numTasks; i++)
	{
		potentialMakespans[DiscreteCoord(matching[i])] += hETCMatrix[(i * numMachines) + DiscreteCoord(matching[i])];
	}

	//When we are deciding on the best fitness, we consider the makespan of all particles and choose the highest.
	//All other makespans are reduced to be proportions of this highest makespan. This allows us to express both
	//makespan and energy use in a manner that is equivalent to one another. (max makespan versus max energy use)
	makespan = potentialMakespans[0];

	//The fitness of the Particle is the makespan of this job-machine configuration.
	for (i = 1; i < numMachines; i++)
	{
		if (makespan < potentialMakespans[i])
			makespan = potentialMakespans[i];
	}

	free(potentialMakespans);

	return makespan;
}

/* ComputeEnergyUse
 *
 * Computes the total energy use required by the given
 * task - machine matching.
 */
float ComputeEnergyUse(float *matching, int numTasks)
{
	int i;
	float energyUse;

	energyUse = 0.0f;

	//Our total energy use is equal to the amount of time a task will take on the machine it is mapped to multiplied by the energy
	//per unit of time this machine requires when under load. So, we first generate our matchi
	for (i = 0; i < numTasks; i++)
	{
		energyUse += hETCMatrix[(i * numMachines) + DiscreteCoord(matching[i])] * hMachines[DiscreteCoord(matching[i])].energy;
	}

	return energyUse;
}

