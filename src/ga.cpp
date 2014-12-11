#include "ga.h"

inline float randFloat()
{
  return ( (float)(rand()) + 1. ) / ( (float)(RAND_MAX) + 1. );
}

inline float randRange(float Min, float Max)
{
  return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

bool compareIndividual(individual a, individual b)
{
  return a.fitness < b.fitness;
}

genetic::genetic(ga_params myParams):
  myParams(myParams),
  myEngine(random_device()()),
  myNormal(0.0, myParams.mutAmount)
{
  for (int i = 0; i < myParams.popSize; i++)
  {
    individual newIndividual;
    newIndividual.fitness = 1e-12;
    newIndividual.genome = new float[myParams.genomeSize];
    for (int j = 0; j < myParams.genomeSize; j++)
    {
      newIndividual.genome[j] = randRange(-myParams.initRange, myParams.initRange);
    }
    population.push_back(newIndividual);
  }

  numGen = 0;
  numEval = 0;
  chunkSize = 500;
}

void genetic::mutate(individual a)
{
  for (int i = 0; i < myParams.genomeSize; i++)
  {
    if (randFloat() < myParams.mutRate)
    {
      a.genome[i] += myNormal(myEngine);
    }
  }
}

void genetic::crossOver(individual a, individual b)
{
  for (int i = 0; i < myParams.genomeSize; i++)
  {
    if (randFloat() < myParams.crossRate)
    {
      float temp = b.genome[i];
      b.genome[i] = a.genome[i];
      a.genome[i] = temp;
    }
  }
}

void genetic::getStats()
{
  totalFitness = 0.0;
  minFitness = FLT_MAX;
  maxFitness = FLT_MIN;
  for (int i = 0; i < myParams.popSize; i++)
  {
    float f = population[i].fitness;
    totalFitness += f;

    if (f < minFitness)
    {
      minFitness = f;
    }
    if (f > maxFitness)
    {
      maxFitness = f;
    }
  }
  meanFitness = totalFitness / myParams.popSize;
}

int genetic::fitnessSelection()
{
  int i;
  while (true)
  {
    i = rand() % myParams.popSize;
    if (randFloat() < population[i].fitness / totalFitness)
    {
      return i;
    }
  }
}

/* only call this if population is already sorted in ascending order! */
void genetic::linearRanking()
{
  for (int i = 0; i < myParams.popSize; i++)
  {
    population[i].fitness = i + 1.0;
  }
}

void genetic::step()
{
  sort(population.begin(), population.end(), compareIndividual);
  linearRanking();
  getStats();

  vector<individual> newPopulation;

  for (int i = 0; i < myParams.numToReplace; i++)
  {
    int a_i = fitnessSelection();
    int b_i = fitnessSelection();

    individual a, b;
    copyIndividual(population[a_i], a);
    copyIndividual(population[b_i], b);

    crossOver(a, b);
    mutate(a); mutate(b);

    newPopulation.push_back(a);
    newPopulation.push_back(b);
  }

  for (int i = 0; i < myParams.popSize; i++)
  {
    if (i < myParams.numToReplace)
    {
      deleteIndividual(population[i]);
    }
    else
    {
      newPopulation.push_back(population[i]);
    }
  }

  population.clear();
  population = newPopulation;
  numGen++;
}

genetic::~genetic()
{
  for (int i = 0; i < myParams.popSize - 1; i++)
  {
    individual a = population[i];
    delete[] a.genome;
  }
}