#ifndef GA
#define GA

#include <algorithm>
#include <vector> 
#include <random>
#include <iostream>
#include "float.h"

using namespace std;

struct ga_params
{
  float mutRate;
  float mutAmount;
  float crossRate;
  int popSize;
  int genomeSize;
  int numToReplace;
  float initRange;
};

struct individual
{
  float *genome;
  float fitness;
};

class genetic
{
private:
  int numGen, numEval, chunkSize;
  float totalFitness, maxFitness, minFitness, meanFitness;

  ga_params myParams;
  vector<individual> population;

  minstd_rand myEngine;
  normal_distribution<float> myNormal;

public:
  genetic(ga_params myParams);
  ~genetic();

  int fitnessSelection();
  void mutate(individual a);
  void crossOver(individual a, individual b);

  void linearRanking();
  void getStats();
  void step();

  //copy individual b to a
  inline void copyIndividual(individual a, individual b)
  {
    a.genome = new float[myParams.genomeSize];
    copy(b.genome, b.genome + myParams.genomeSize, a.genome);
    a.fitness = b.fitness;
  }

  inline void deleteIndividual(individual a)
  {
    delete[] a.genome;
    a.genome = NULL;
  }

  inline float *getGenome(int i)
  {
    return population[i].genome;
  }

  inline void setFitness(int i, float fitness)
  {
    numEval++;
    population[i].fitness = fitness;
  }

  inline void printStats()
  {
    cout << "gen: " << numGen << " eval: " << numEval << " maxFitness: " << maxFitness << " meanFitness: " << meanFitness << " minFitness: " << minFitness << endl;
  }

};


#endif