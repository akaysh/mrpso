#ifndef _HELPER_PSO_H_
#define _HELPER_PSO_H_

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

extern float *hETCMatrix;
extern Machine *hMachines;
extern Task *hTasks;

Machine* BuildMachineList(char *filename);
Task* BuildTaskList(char *filename);
float* GeneratehETCMatrix();

void FreeMemory();

void PrinthETCMatrix();
void PrinthMachines();
void PrinthTasks();

int GetnumMachines();
int GetnumTasks();

float ComputeMakespan(float *matching, int numTasks);
float ComputeEnergyUse(float *matching, int numTasks);

#endif
