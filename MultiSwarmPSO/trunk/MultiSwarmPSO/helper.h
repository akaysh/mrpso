#ifndef _HELPER_PSO_H_
#define _HELPER_PSO_H_

#define RECORD_VALUES

#include <cuda_runtime.h>

#define MAX_RAND_GEN 419430400

typedef enum VELOCITY_UPDATE_STYLE
{
	BASIC,		//Continuous PSO
	DISCRETE	//Discrete PSO
} velocity_update_style;

/* Particle
 *
 * Maintains all of the data required by a Particle in the PSO algorithm.
 * Allows for movement across as many dimensions as necessary.
 */
typedef struct
{
	float fitness; //The current fitness of this Particle.
	float pBest;   //The best fitness of this Particle.

	float *positionVector;		//The current location of this Particle in each dimension.
	float *pBestPositionVector;	//The location where this Particle found pBest.
	float *velocityVector;		//The current velocity of this Particle in each dimension.
} Particle;

typedef float4 ArgStruct;

/* Swarm
 *
 * A Swarm is composed of a series of Particles as well as a global best position/value.
 */
typedef struct 
{
	Particle *particles;		//The array of Particles composing this swarm.
	float gBest;				//The fitness of the global best position.
	float *gBestPositionVector; //The actual global best position for this swarm.
} Swarm;

/* Task
 *
 * Simple structure for representing a Task. Currently only manages the task
 * length (in millions of instructions).
 */
typedef struct
{
	float length;		//Amount of work (in terms of millions of instructions) required by this task.
} Task;

/* Machine
 *
 * A structure for maintaining the properties of a Machine.
 */
typedef struct
{
	float MIPS;			//MIPS-rating of this machine.
	float energy;		//Amount of energy used by this Machine every unit of time.
} Machine;

/* RunConfiguration
 *
 * A structure for storing the configuration required for a single run of
 * the PSO algorithm.
 */
typedef struct  
{
	char taskFile[512];			//The file containing task definitions.
	char machineFile[512];		//The file containing machine definitions.
	int numSwarms;				//The number of swarms to use for a given objective.
	int numParticles;			//The number of particles per swarm.
	int numIterations;			//The total number of iterations.
	int iterationsBeforeSwap;	//The number of iterations before swarms communicate with one another.
	int numParticlesToSwap;		//The number of particles to swap between swarms when communicating.
	float w;					//The inertial weight parameter.
	float wDecay;				//The amount of decay in w at each iteration.
	float c1;					//The local perturbation parameter.
	float c2;					//The global perturbation parameter.
	int threadsPerBlock;		//The number of threads per block to use for basic kernels.
	int numTests;				//The number of tests to average the results by.
} RunConfiguration;

extern float *hETCMatrix;
extern Machine *hMachines;
extern Task *hTasks;

int OpenRunsFile(char *filename);
RunConfiguration *GetNextRun();
void CloseRunsFile();

Machine* BuildMachineList(char *filename);
Task* BuildTaskList(char *filename);
float* GenerateETCMatrix();

void FreeCPUMemory();

void PrintETCMatrix();
void PrintMachines();
void PrintTasks();
void PrintTestResults(int passed);

int GetNumMachines();
int GetNumTasks();

int DiscreteCoord(float value);

float ComputeMakespan(float *matching, int numTasks);
float ComputeEnergyUse(float *matching, int numTasks);

int CalcNumBlocks(int numThreads, int threadsPerBlock);
bool InitCUDA();

void AllocateGPUMemory(RunConfiguration *run);
void FreeGPUMemory();
void GenerateRandsGPU(int total, float *deviceMem);
void InitRandsGPU();
void FreeRandsGPU();
void GenRandsGPU(int numToGen, float *deviceMem);

extern float *hPosition, *dPosition;

extern float *hVelocity, *dVelocity;

extern float *hPBest, *dPBest, *hPBestPosition, *dPBestPosition;

extern float *hGBest, *dGBest, *hGBestPosition, *dGBestPosition;

extern float *dScratch;

extern int *dBestSwapIndices, *dWorstSwapIndices;

extern float *hRands, *dRands;

extern float *hFitness, *dFitness;

extern int initializationRandCount, iterationRandCount;

#endif
