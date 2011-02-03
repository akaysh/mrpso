/* swarm.c
 *
 * Implements a single Swarm as described by the
 * basic Particle Swarm Optimization concept.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "helper.h"
#include "cpu_pso.h"

void InitializePSO(Particle *particles, float *gBest, float *gBestPositionVector, int numParticles, int numTasks, int numMachines, float w, float c1, float c2, velocity_update_style vStyle, int isMakespan)
{
	int i;

	//Initialize the components of the swarm.
	InitializeSwarm(particles, numParticles, numTasks, numMachines);

	//Update the fitness based on the starting location of each Particle
	for (i = 0; i < numParticles; i++)
	{
		if (isMakespan)
			UpdateFitnessMakespan(&particles[i], numTasks, numMachines);
		else
			UpdateFitnessEnergy(&particles[i], numTasks, numMachines);

		particles[i].pBest = particles[i].fitness;
	}

	//Set the global best to the first particle for now...
	*gBest = particles[0].fitness;

	for (i = 0; i < numTasks; i++)
		gBestPositionVector[i] = particles[0].positionVector[i];

	//Determine the current global best value...
	FindGlobalBest(particles, numParticles, gBest, gBestPositionVector, numTasks);

	//Finally, compute the initial velocity of the particles
	switch (vStyle)
	{
		case BASIC:
			UpdateVelocitiesBasic(particles, numParticles, numTasks, numMachines, w, c1, c2, gBestPositionVector);
			break;
		case DISCRETE:
			UpdateVelocitiesDiscrete(particles, numParticles, numTasks, numMachines);
			break;
	}	
}

/* InitializeSwarm
 *
 * Initializes the Particles in the swarm and positions
 * them at random places in the solution space.
 */
void InitializeSwarm(Particle *particles, int numParticles, int numTasks, int numMachines)
{
	int i, j;
	int randMax = numMachines - 1;

	srand((unsigned int) time(NULL));

	for (i = 0; i < numParticles; i++)
	{
		particles[i].positionVector = (float *) malloc(numTasks * sizeof(float));
		particles[i].velocityVector = (float *) calloc(numTasks, sizeof(float));
		particles[i].pBestPositionVector = (float *) malloc(numTasks * sizeof(float));
		particles[i].fitness = 0.0f;
		particles[i].pBest = 0.0f;

		//Give the Particle a random starting position in the solution space as well as
		//a random starting velocity.
		for (j = 0; j < numTasks; j++)
		{
			particles[i].positionVector[j] = (randMax * ((float) rand() / (float) RAND_MAX));
			particles[i].pBestPositionVector[j] = particles[i].positionVector[j];

			particles[i].velocityVector[j] = ((randMax * 2) * (float) rand() / (float) RAND_MAX) - randMax;
		}		
	}	
}

/* UpdateFitnessEnergy
 *
 * Updates the fitness of a given Particle taking into account
 * only the energy use.
 */
float UpdateFitnessEnergy(Particle *particle, int numTasks, int numMachines)
{
	int i;
	float energyUse;

	energyUse = 0.0f;

	//Our total energy use is equal to the amount of time a task will take on the machine it is mapped to multiplied by the energy
	//per unit of time this machine requires when under load. So, we first generate our matchi
	for (i = 0; i < numTasks; i++)
	{
		energyUse += hETCMatrix[(i * numMachines) + DiscreteCoord(particle->positionVector[i])] * hMachines[DiscreteCoord(particle->positionVector[i])].energy;
	}

	particle->fitness = energyUse;

	//Check to see if the local best needs updating.
	if (particle->fitness < particle->pBest)
	{
		particle->pBest = particle->fitness;

		for (i = 0; i < numTasks; i++)
			particle->pBestPositionVector[i] = particle->positionVector[i];
	}

	return energyUse;
}

/* UpdateFitnessMakespan
 *
 * Updates the fitness of a given Particle taking into account
 * only the makespan.
 */
float UpdateFitnessMakespan(Particle *particle, int numTasks, int numMachines)
{
	int i;
	float *matching;
	float makespan;

	matching = (float *) calloc(numMachines, sizeof(float));

	//Each dimension of the Particle's position vector represents a task, the actual
	//position in that dimension represents the machine this task is matched to.
	for (i = 0; i < numTasks; i++)
	{
		matching[DiscreteCoord(particle->positionVector[i])] += hETCMatrix[(i * numMachines) + DiscreteCoord(particle->positionVector[i])];
	}

	//When we are deciding on the best fitness, we consider the makespan of all particles and choose the highest.
	//All other makespans are reduced to be proportions of this highest makespan. This allows us to express both
	//makespan and energy use in a manner that is equivalent to one another. (max makespan versus max energy use)
	makespan = matching[0];

	//The fitness of the Particle is the makespan of this job-machine configuration.
	for (i = 1; i < numMachines; i++)
	{
		if (makespan < matching[i])
			makespan = matching[i];
	}

	//The makespan is equal to the largest of the largest ECT values for each job on each machine.
	particle->fitness = makespan;

	//Check to see if the local best needs updating.
	if (particle->fitness < particle->pBest)
	{
		particle->pBest = particle->fitness;

		for (i = 0; i < numTasks; i++)
			particle->pBestPositionVector[i] = particle->positionVector[i];
	}

	free(matching);

	return makespan;
}

/* UpdateFitnesses
 *
 * Updates the fitness values for all of the Particles in the swarm.
 */
void UpdateFitnesses(Particle *particles, int numParticles, int numTasks, int numMachines, int isMakespan)
{
	int i;

	float (*fitnessFunc) (Particle *, int, int) = NULL;

	if (isMakespan)
		fitnessFunc = &UpdateFitnessMakespan;
	else
		fitnessFunc = &UpdateFitnessEnergy;

	for (i = 0; i < numParticles; i++)
	{
		fitnessFunc(&particles[i], numTasks, numMachines);
	}
}

/* FindGlobalBest
 *
 * Updates the global best value and position for the given swarm if necessary.
 */
void FindGlobalBest(Particle *particles, int numParticles, float *currGBest, float *gBestPositionVector, int numTasks)
{
	int i, particleIndex;

	particleIndex = -1;

	//Search for a new global best if one exists.
	for (i = 0; i < numParticles; i++)
	{
		if (particles[i].pBest < *currGBest)
		{
			*currGBest = particles[i].pBest;
			particleIndex = i;
		}
	}

	//If we found a new global best, copy the position over from it.
	//We perform this step separate from the search as we don't want to
	//continually copy data if we find multiple global "bests" in the 
	//previous for loop.
	if (particleIndex >= 0)
	{
		for (i = 0; i < numTasks; i++)
			gBestPositionVector[i] = particles[particleIndex].pBestPositionVector[i];
	}
}


/* UpdatePosition
 *
 * Updates the current position of the given Particle based on
 * its current velocity. Ensures that the Particle cannot go outside
 * of the solution space bounds.
 */
void UpdatePositionBasic(Particle *particle, int numTasks, int numMachines)
{
	int i;

	//Move the Particle (ensuring that we clamp the maximum or minimum position)
	for (i = 0; i < numTasks; i++)
	{
		particle->positionVector[i] += particle->velocityVector[i];

		if (particle->positionVector[i] < 0)
			particle->positionVector[i] = 0.0f;
		else if (particle->positionVector[i] > numMachines - 1)
			particle->positionVector[i] = (float) numMachines - 1;
	}
}

/* F
 *
 * The 'F' function used in the Discrete PSO method described
 * in Kang, et al.
 */
int F(int pBestPos, int positional)
{
	int finalVal;
	int difference;

	difference = pBestPos - positional;

	if (difference > 0)
		finalVal = positional + (rand() % difference);
	else
		finalVal = positional;

	return finalVal;
}

/* UpdatePositionDiscrete
 *
 * Updates the position of the given Particle using the Discrete PSO method described in
 * Kang, et al. 
 */
void UpdatePositionDiscrete(Particle *particle, float w, float *gBestPositionVector, int numTasks)
{
	int i;
	int positional, personal, global;
	float randNum;

	for (i = 0; i < numTasks; i++)
	{
		//Update the first component, the positional perturbation (Xi (+) w (x) Vi)
		randNum = (float) rand() / (float) RAND_MAX;

		if (randNum < w)
			positional = (int) particle->velocityVector[i];
		else
			positional = (int) particle->positionVector[i];

		//Update the second component, the personal cognition, F(pBest, positional)
		personal = F( (int) particle->pBestPositionVector[i], positional);

		//Update the third component, the global cognition, F(gBest, personal)
		//As this is composed of all the other components as well, this defines our
		//new position in this dimension.
		global = F( (int) gBestPositionVector[i], personal);

		particle->positionVector[i] = (float) global;
	}
}

/* UpdatePositionsBasic
 *
 * Updates the position of each Particle in the given swarm using continuous PSO.
 */
void UpdatePositionsBasic(Particle *particles, int numParticles, int numTasks, int numMachines)
{
	int i;
	
	for (i = 0; i < numParticles; i++)
		UpdatePositionBasic(&particles[i], numTasks, numMachines);
}

/* UpdatePositionsDiscrete
 *
 * Updates the position of each Particle in the given swarm using the discrete PSO
 * method described in Kang, et al.
 */
void UpdatePositionsDiscrete(Particle *particles, int numParticles, int numTasks, float w, float *gBestPositionVector)
{
	int i;
	
	for (i = 0; i < numParticles; i++)
		UpdatePositionDiscrete(&particles[i], w, gBestPositionVector, numTasks);
}

/* UpdateVelocityBasic
 *
 * Updates the velocity of the given Particle using the basic, traditional PSO form. 
 * That is, the velocity is updated using the traditional method with a continuous
 * domain (which will be converted to a discrete domain using the GenDiscreteCoord method when used).
 */
void UpdateVelocityBasic(Particle *particle, float w, float c1, float c2, float *gBestPositionVector, int numTasks, int numMachines)
{
	int i;
	float newVelocity;
	
	//Update the velocity for each dimension of the Particle using the traditional rule:
	//Vi = w * Vi + c1 * rand() * (pbesti - Xi) + c2 * rand() * (gbesti - Xi)
	for (i = 0; i < numTasks; i++)
	{
		newVelocity = w * particle->velocityVector[i];
		newVelocity += c1 * ( ((float) rand() / (float) RAND_MAX) * (particle->pBestPositionVector[i] - particle->positionVector[i]) );
		newVelocity += c2 * ( ((float) rand() / (float) RAND_MAX) * (gBestPositionVector[i] - particle->positionVector[i]) );

		if (newVelocity > numMachines * .5f)
			newVelocity = numMachines * .5f;
		else if (newVelocity < numMachines * -0.5f)
			newVelocity = numMachines * -0.5f;

		particle->velocityVector[i] = newVelocity;
	}
}

/* UpdateVelocityDiscrete
 *
 * Updates the velocity of the given Particle using the Discrete PSO method described in
 * Kang, et al. Simply put, this method generates a random integer between 0 and MAX.
 * This value is later used in the position update function and may or may not impact the
 * actual position of the Particle.
 */
void UpdateVelocityDiscrete(Particle *particle, int numTasks, int numMachines)
{
	int i;

	for (i = 0; i < numTasks; i++)
		particle->velocityVector[i] = (float) (rand() % numMachines);

}

/* UpdateVelocitiesDiscrete
 *
 * Updates the velocities of each Particle in the given swarm using the basic
 * continuous PSO method by Shi and Eberhart.
 */
void UpdateVelocitiesBasic(Particle *particles, int numParticles, int numTasks, int numMachines, float w, float c1, float c2, float *gBestPositionVector)
{
	int i;
	
	for (i = 0; i < numParticles; i++)
		UpdateVelocityBasic(&particles[i], w, c1, c2, gBestPositionVector, numTasks, numMachines);
}

/* UpdateVelocitiesDiscrete
 *
 * Updates the velocities of each Particle in the given swarm using the discrete PSO
 * method described in Kang, et al.
 */
void UpdateVelocitiesDiscrete(Particle *particles, int numParticles, int numTasks, int numMachines)
{
	int i;
	
	for (i = 0; i < numParticles; i++)
		UpdateVelocityDiscrete(&particles[i], numTasks, numMachines);
}


