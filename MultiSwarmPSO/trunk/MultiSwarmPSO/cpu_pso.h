#ifndef _CPU_PSO_H_
#define _CPU_PSO_H_

#include "helper.h"

float* RunMakespanPSO(int numParticles, int numTasks, int numMachines, float w, float wDecay, float c1, float c2, int numIterations, velocity_update_style vStyle);
float* RunEnergyPSO(int numParticles, int numTasks, int numMachines, float w, float wDecay, float c1, float c2, int numIterations, velocity_update_style vStyle);

void InitializePSO(Particle *particles, float *gBest, float *gBestPositionVector, int numParticles, int numTasks, int numMachines, float w, float c1, float c2, velocity_update_style vStyle, int isMakespan);
void UpdateFitnesses(Particle *particles, int numParticles, int numTasks, int numMachines, int isMakespan);

void UpdateVelocityBasic(Particle *particle, float w, float c1, float c2, float *gBestPositionVector, int numTasks, int numMachines);
void UpdatePositionBasic(Particle *particle, int numTasks, int numMachines);

void InitializeSwarm(Particle *particles, int numParticles, int numTasks, int numMachines);
void FindGlobalBest(Particle *particles, int numParticles, float *currGBest, float *gBestPositionVector, int numTasks);

void UpdatePositionsBasic(Particle *particles, int numParticles, int numTasks, int numMachines);
void UpdatePositionsDiscrete(Particle *particles, int numParticles, int numTasks, float w, float *gBestPositionVector);

void UpdateVelocitiesBasic(Particle *particles, int numParticles, int numTasks, int numMachines, float w, float c1, float c2, float *gBestPositionVector);
void UpdateVelocitiesDiscrete(Particle *particles, int numParticles, int numTasks, int numMachines);

float* RunVEPSO(int numParticles, int numTasks, int numMachines, float w, float wDecay, float c1, float c2, int numIterations, velocity_update_style vStyle, int doRandomSwap);
float UpdateFitnessMakespan(Particle *particle, int numTasks, int numMachines);
float UpdateFitnessEnergy(Particle *particle, int numTasks, int numMachines);

void RunMRPSO(RunConfiguration *run);
float *GetRecordedData();

int CompareParticles(const void *elem1, const void *elem2);

#endif

