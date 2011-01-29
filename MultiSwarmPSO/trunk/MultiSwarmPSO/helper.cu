#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda.h"
#include "curand.h"
#include "helper.h"
#include "gpu_pso.h"

float *hPosition, *dPosition;

float *hVelocity, *dVelocity;

float *hPBest, *dPBest, *hPBestPosition, *dPBestPosition;

float *hGBest, *dGBest, *hGBestPosition, *dGBestPosition;

float *dScratch;

int *dBestSwapIndices, *dWorstSwapIndices;

float *hRands, *dRands;

float *hFitness, *dFitness;

int numMachines, numTasks;

float *hETCMatrix = NULL;
Machine *hMachines = NULL;
Task *hTasks = NULL;

FILE *runsFile;
RunConfiguration *run = NULL;

int initializationRandCount, iterationRandCount;

float initTime, swapTime, determineSwapTime, findBestsTime, updatePosVelTime, fitnessTime;

curandGenerator_t randGenGPU;

void ResetTimers()
{
	initTime = 0.0f;
	swapTime = 0.0f;
	determineSwapTime = 0.0f;
	findBestsTime = 0.0f;
	updatePosVelTime = 0.0f;
	fitnessTime = 0.0f;
}

/* AllocateGPUMemory
 *
 * Allocates the required memory on the GPU's global memory system.
 */
void AllocateGPUMemory(RunConfiguration *run)
{
	//initializationRandCount = run->numSwarms * run->numParticles * numTasks * 2;
	//iterationRandCount = run->numParticles * run->numSwarms * numTasks * run->numIterations * 2;
	cudaMalloc((void**) &dPosition, run->numParticles * run->numSwarms * numTasks * sizeof(float));
	cudaMalloc((void**) &dVelocity, run->numParticles * run->numSwarms * numTasks * sizeof(float));
	cudaMalloc((void**) &dFitness, run->numParticles * run->numSwarms * sizeof(float));
	cudaMalloc((void**) &dGBest, run->numSwarms * sizeof(float));
	cudaMalloc((void**) &dGBestPosition, run->numSwarms * numTasks * sizeof(float));
	cudaMalloc((void**) &dPBest, run->numParticles * run->numSwarms * sizeof(float));
	cudaMalloc((void**) &dPBestPosition, run->numParticles * run->numSwarms * numTasks * sizeof(float));
	cudaMalloc((void**) &dScratch, run->numParticles * run->numSwarms * numMachines * sizeof(float));
	cudaMalloc((void**) &dBestSwapIndices, run->numSwarms * run->numParticlesToSwap * sizeof(int));
	cudaMalloc((void**) &dWorstSwapIndices, run->numSwarms * run->numParticlesToSwap * sizeof(int));
	cudaMalloc((void**) &dRands, MAX_RAND_GEN * sizeof(float));
}

/* FreeGPUMemory
 *
 * Frees the previously allocated GPU memory.
 */
void FreeGPUMemory()
{
	cudaFree(dPosition);
	cudaFree(dVelocity);
	cudaFree(dFitness);
	cudaFree(dGBest);
	cudaFree(dGBestPosition);
	cudaFree(dPBest);
	cudaFree(dPBestPosition);
	cudaFree(dScratch);
	cudaFree(dBestSwapIndices);
	cudaFree(dWorstSwapIndices);
	cudaFree(dRands);
}

void InitRandsGPU()
{
	unsigned int free, total;

	cuMemGetInfo(&free, &total);
	printf("Free: %d, total: %d\n", free, total);

	curandCreateGenerator(&randGenGPU, CURAND_RNG_PSEUDO_XORWOW);
	printf((cudaGetErrorString(cudaGetLastError())));

	cuMemGetInfo(&free, &total);
	printf("Free: %d, total: %d\n", free, total);
	curandSetPseudoRandomGeneratorSeed(randGenGPU, (unsigned int) time(NULL));
	curandGenerateSeeds(randGenGPU);
	cudaThreadSetLimit(cudaLimitStackSize, 1024);
}

void FreeRandsGPU()
{
	curandDestroyGenerator(randGenGPU);

	//Reset the stack size to get our memory back for sm_20 until this bug is fixed.
	cudaThreadSetLimit(cudaLimitStackSize, 1024);
}

void GenRandsGPU(int numToGen, float *deviceMem)
{
	curandGenerateUniform(randGenGPU, deviceMem, numToGen);
}

/* GenerateRandsGPU
 *
 * Generates all of the GPU random numbers required for the
 * given run configuration.
 *
 * For legacy purposes (unit tests).
 */
void GenerateRandsGPU(int total, float *deviceMem)
{
	int numRands;
	curandGenerator_t gen1;

	numRands = total > 0 ? total : initializationRandCount + iterationRandCount;

	curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen1, (unsigned int) time(NULL));
	curandGenerateUniform(gen1, deviceMem, numRands);
	curandDestroyGenerator(gen1);

	//Reset the stack size to get our memory back on fermi-based GPUs
	cudaThreadSetLimit(cudaLimitStackSize, 1024);
}

/* OpenRunsFile
 *
 * Opens the requested file containing run data.
 * Returns true if the file was opened, false if otherwise.
 */
int OpenRunsFile(char *filename)
{
	runsFile = fopen(filename, "r");
	
	return runsFile == NULL ? 0 : 1;
}

/* LoadRunConfig
 *
 * Loads the run configuration described in the provided line and
 * returns a RunConfiguration struct populated with the necessary information.
 */
RunConfiguration *LoadRunConfig(char *line)
{
	if (line != NULL)
	{
		sscanf(line, "%s %s %d %d %d %f %f %f %f %d %d %d %d", &run->taskFile, &run->machineFile, &run->numSwarms, &run->numParticles, 
			   &run->numIterations, &run->w, &run->wDecay, &run->c1, &run->c2, &run->iterationsBeforeSwap, &run->numParticlesToSwap, &run->threadsPerBlock, 
			   &run->numTests);
	}

	return run;
}

/* RunConfiguration
 *
 * Gets the next run defined in the run file.
 * Returns NULL if no further runs have been defined or if
 * the input file has not been opened via OpenRunsFile()
 */
RunConfiguration *GetNextRun()
{
	int done = 0;
	char line[512];

	if (run == NULL)
		run = (RunConfiguration *) malloc(sizeof(RunConfiguration));

	if (runsFile != NULL)
	{
		while (!done && fgets(line, sizeof line, runsFile) != NULL)
		{
			if (line[0] != '#')
				done = true;
		}

		if (!done && feof(runsFile))
		{
			run->numSwarms = -1;
		}
		else
		{
			LoadRunConfig(line);
		}
	}
	else
	{
		run->numSwarms = -1;
	}

	return run;
}

/* CloseRunsFile
 * 
 * Closes the file containing run definitions if it has
 * been opened and frees the corresponding memory.
 */
void CloseRunsFile()
{
	if (runsFile != NULL)
		fclose(runsFile);

	if (run != NULL)
	{
		free(run);
		run = NULL;
	}
}

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

/* FreeCPUMemory
 *
 * Frees the memory allocated by building the Machine and Task
 * lists as well as the ETC Matrix.
 */
void FreeCPUMemory()
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

/* DiscreteCoord
 *
 * Generates a discrete coordinate from a continuous coordinate.
 */
int DiscreteCoord(float value)
{
	return (int) floor(value);
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

int CalcNumBlocks(int numThreads, int threadsPerBlock)
{
	int numBlocks;

	if (numThreads % threadsPerBlock == 0)
		numBlocks = numThreads / threadsPerBlock;
	else
		numBlocks = (numThreads / threadsPerBlock) + 1;
	numBlocks = numBlocks == 0 ? 1 : numBlocks;

	return numBlocks;
}

#if __DEVICE_EMULATION__

bool InitCUDA() { return true; }

#else
bool InitCUDA()
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) 
	{
		fprintf(stderr, "A CUDA-capable device was not detected.\n");
		return false;
	}

	for(i = 0; i < count; i++) 
	{
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
		{
			if(prop.major >= 1) 
				break;			
		}
	}

	if(i == count) 
	{
		fprintf(stderr, "A CUDA-capable device was not detected.\n");
		return false;
	}

	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}
#endif

void PrintTestResults(int passed)
{
	if (passed)
		printf("\t[SUCCESS] Test passed!\n\n");
	else
		printf("\t[FAILURE] Test failed!\n\n");
}
