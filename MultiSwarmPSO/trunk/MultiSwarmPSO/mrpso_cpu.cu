#include <stdio.h>
#include "helper.h"
#include "cpu_pso.h"

#define THRESHOLD 0.00001

#ifdef RECORD_VALUES
	float *gBestValues = NULL;
#endif

int CompareParticles(const void *elem1, const void *elem2)
{
	if ( ((const Particle *) elem1)-> fitness < ((const Particle *) elem2)->fitness)
		return -1;

	return ( ((const Particle *) elem1)-> fitness > ((const Particle *) elem2)->fitness);
}

/* SwapParticles
 *
 * Swaps the n best Particles of a swarm with its neighbour's n worst Particles.
 * Uses a simple insertion sort to sort as the number of particles per swarm is small enough
 * that algorithms such as quicksort or merge sort cannot provide any performance benefits.
 */
void SwapParticles(Particle *particles, int numToSwap, int numSwarms, int numParticles)
{
	int i, j;	
	int swarmToSwapTo;
	Particle tempParticle;

	//First, we need to sort each of the swarm's particles.
	for (i = 0; i < numSwarms; i++)
		qsort(&particles[numParticles * i], numParticles, sizeof(Particle), CompareParticles);

	//Every swarm, i, swaps 5 of its best particles with 5 of the worst particles from swarm (i + 1) % numSwarms
	for (i = 0; i < numSwarms; i++)
	{
		swarmToSwapTo = (i + 1) % numSwarms;

		//Grab the n worst particles from our neighbor, and give the best n in return.
		for (j = 0; j < numToSwap; j++)
		{
			tempParticle = particles[(numParticles * swarmToSwapTo) + (numParticles - j - 1)];
			particles[(numParticles * swarmToSwapTo) + (numParticles - j - 1)] = particles[(numParticles * i) + j];
			particles[(numParticles * i) + j] = tempParticle;
		}
	}
}

void UpdateSwarmAttributes(Particle *particles, float *gBest, float *gBestPositionVector, int numParticles, int numTasks, int numMachines, float w, float c1, float c2)
{
	int i;

	//Update the global best first to ensure that we capture the information from the
	//previous iteration's swap immediately.
	FindGlobalBest(particles, numParticles, gBest, gBestPositionVector, numTasks);

	//Update the velocity, position, and fitness of each Particle in this swarm.
	for (i = 0; i < numParticles; i++)
	{
		UpdateVelocityBasic(&particles[i], w, c1, c2, gBestPositionVector, numTasks, numMachines);
		UpdatePositionBasic(&particles[i], numTasks, numMachines);
		UpdateFitnessMakespan(&particles[i], numTasks, numMachines);
	}
}

void UpdateAllSwarmsAttributes(Particle *particles, float *gBest, float *gBestPositionVector, int numSwarms, int numParticles, int numTasks, int numMachines, float w, float c1, float c2)
{
	int i;

	//Update the Particles for each swarm
	for (i = 0; i < numSwarms; i++)
	{
		UpdateSwarmAttributes(&particles[numParticles * i], &gBest[i], &gBestPositionVector[numTasks * i], numParticles, numTasks, numMachines, w, c1, c2);
	}
}

void RunMRPSO(RunConfiguration *run)
{
	int i;
	Particle *particles;	//This will store the particles for all of our swarms
	float *gBest, *gBestPositionVector;
	int numTasks = GetNumTasks();
	int numMachines = GetNumMachines();

	particles = (Particle *) malloc(run->numSwarms * run->numParticles * sizeof(Particle));
	gBest = (float *) malloc(run->numSwarms * sizeof(float));
	gBestPositionVector = (float *) malloc(run->numSwarms * numTasks * sizeof(float));

#ifdef RECORD_VALUES
	if (gBestValues != NULL)
		free(gBestValues);

	gBestValues = (float *) malloc(run->numSwarms * run->numIterations * sizeof(float));
#endif

	//Initialize all of the swarms
	for (i = 0; i < run->numSwarms; i++)
		InitializePSO(&particles[run->numParticles * i] , &gBest[i], &gBestPositionVector[numTasks * i], run->numParticles, numTasks, numMachines, 
					  run->w, run->c1, run->c2, BASIC, 1);

	//Now continually iterate until we've reached our maximum quota of iterations...
	for (i = 0; i < run->numIterations; i++)
	{
		if (run->w > 0.4)
			run->w *= run->wDecay;

		//Update the velocity, position, and fitness of each swarm's particles.
		UpdateAllSwarmsAttributes(particles, gBest, gBestPositionVector, run->numSwarms, run->numParticles, GetNumTasks(), GetNumMachines(), run->w, run->c1, run->c2);

#ifdef RECORD_VALUES
		for (int j = 0; j < run->numSwarms; j++)
			gBestValues[(run->numSwarms * i) + j] = gBest[j];
#endif

		//If we've reached the swap threshold, swap particles between swarms.
		if (i % run->iterationsBeforeSwap == 0)
			SwapParticles(particles, run->numParticlesToSwap, run->numSwarms, run->numParticles);
	}

	free(particles);
}

#ifdef RECORD_VALUES
float *GetRecordedData()
{
	return gBestValues;
}
#endif
