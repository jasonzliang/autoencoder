#ifndef GA
#define GA

#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <omp.h>
#include "float.h"
#include <time.h>
#include <climits>

using namespace std;

struct ga_params
{
  float mutRate;
  float mutAmount;
  float crossRate;
  int popSize;
  int genomeSize;
  int numWeights;
  int numToReplace;
  float initRange;

  float alpha = 1.0;
  bool useGradient = false;
  bool zeroMutate = false;
  bool truncateSel = false;
  int chunkSize = 15000;
};

struct individual
{
  float *genome;
  float fitness;
  float scaledFitness;
  int age;
  // int oldRank;
};

class genetic
{
private:
  int numGen, numEval;
  float totalFitness, maxFitness, minFitness, meanFitness, totalScaledFitness;
  int bestIndIndex;

  ga_params myParams;
  vector<individual> population;

  mt19937_64 myEngine;
  cauchy_distribution<float> myDist;
  // normal_distribution<float> myDist;

  int randBankSize;
  float *randBank;
  float *normalBank;

public:
  genetic(ga_params myParams);
  ~genetic();

  int fitnessSelection();
  int truncationSelection();

  void mutate(individual &a);
  void zeroMutate(individual &a);
  void crossOver(individual &a, individual &b);

  void noRanking();
  void linearRanking();
  void powerRanking();
  void sortPopulation();
  void copyIndividual(individual &a, individual &b);
  void getStats();
  void step();

  inline void deleteIndividual(individual &a)
  {
    delete[] a.genome;
    a.genome = NULL;
  }

  inline float *getGenome(int i)
  {
    return population[i].genome;
  }

  // inline int getOldRank(int i)
  // {
  //   return population[i].oldRank;
  // }

  inline void setFitness(int i, float fitness)
  {
    numEval++;
    population[i].fitness = fitness;
    // population[i].fitness = (fitness + population[i].age * population[i].fitness) / (population[i].age + 1);
    population[i].age++;
  }

  inline void printStats()
  {
    cout << "gen: " << numGen << " eval: " << numEval << " maxFitness: " << maxFitness << " realError: " << exp(1.0 / maxFitness) << " meanFitness: " << meanFitness << " minFitness: " << minFitness << endl;
  }

};


#endif