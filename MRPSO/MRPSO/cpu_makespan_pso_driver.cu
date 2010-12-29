#include <stdio.h>
#include <stdlib.h>
#include "cpu_pso.h"
#include "helper.h"

float* RunMakespanPSO(int numParticles, int numTasks, int numMachines, float w, float wDecay, float c1, float c2, int numIterations, velocity_update_style vStyle)
{
	int i;
	Particle *particles;
	float gBest, *gBestPositionVector;
	float *iterationValues;

	particles = (Particle *) malloc(numParticles * sizeof(Particle));
	gBestPositionVector = (float *) malloc(numTasks * sizeof(float));

	iterationValues = (float *) malloc(numIterations * sizeof(float) * 2);

	InitializePSO(particles, &gBest, gBestPositionVector, numParticles, numTasks, numMachines, w, c1, c2, vStyle, 1);

	//Now continually iterate until we've reached our maximum quota of iterations...
	for (i = 0; i < numIterations; i++)
	{		
		if (w > 0.4)
			w *= wDecay;

		//Update the velocity and position...
		switch (vStyle)
		{
			case BASIC:
				UpdateVelocitiesBasic(particles, numParticles, numTasks, numMachines, w, c1, c2, gBestPositionVector);
				UpdatePositionsBasic(particles, numParticles, numTasks, numMachines);
				break;
			case DISCRETE:
				UpdateVelocitiesDiscrete(particles, numParticles, numTasks, numMachines);
				UpdatePositionsDiscrete(particles, numParticles, numTasks, w, gBestPositionVector);
				break;
		}

		//Update Fitnesses
		UpdateFitnesses(particles, numParticles, numTasks, numMachines, 1);

		//Find Global Best
		FindGlobalBest(particles, numParticles, &gBest, gBestPositionVector, numTasks);

		iterationValues[i * 2] = gBest;
		iterationValues[i * 2 + 1] = ComputeEnergyUse(gBestPositionVector, numTasks);
	}
	
	free(particles);
	free(gBestPositionVector);

	return iterationValues;
}