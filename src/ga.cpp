#include "ga.h"

uint64_t s[2];

void seed_xorshift128plus(uint64_t s1, uint64_t s2)
{
  s[0] = s1;
  s[1] = s2;
}

uint64_t xorshift128plus()
{
  uint64_t s1 = s[ 0 ];
  const uint64_t s0 = s[ 1 ];
  s[ 0 ] = s0;
  s1 ^= s1 << 23;
  return ( s[ 1 ] = ( s1 ^ s0 ^ ( s1 >> 17 ) ^ ( s0 >> 26 ) ) ) + s0;
}

inline float randFloat()
{
  return float(xorshift128plus()) / float(ULLONG_MAX);
}

inline float randRange(float Min, float Max)
{
  return ((float(xorshift128plus()) / float(ULLONG_MAX)) * (Max - Min)) + Min;
}

bool compareIndividual(individual a, individual b)
{
  return a.fitness < b.fitness;
}

genetic::genetic(ga_params myParams):
  myParams(myParams),
  myEngine(random_device()()),
  myDist(0.0f, myParams.mutAmount)
{
  cout << "initializing population..." << endl;

  for (int i = 0; i < myParams.popSize; i++)
  {
    individual newIndividual;
    newIndividual.fitness = 0.01;
    newIndividual.age = 0;
    newIndividual.genome = new float[myParams.genomeSize];

    #pragma omp parallel
    {
      seed_xorshift128plus((uint64_t) random_device()(), (uint64_t) random_device()());
      #pragma omp for schedule(static, myParams.chunkSize)
      for (int j = 0; j < myParams.numWeights; j++)
      {
        newIndividual.genome[j] = randRange(-myParams.initRange, myParams.initRange);
      }
    }

    for (int j = myParams.numWeights; j < myParams.genomeSize; j++)
    {
      newIndividual.genome[j] = 0.0f;
    }

    population.push_back(newIndividual);
  }

  numGen = 0;
  numEval = 0;

  cout << "generating random bank..." << endl;
  // randBankSize = min(50 * myParams.genomeSize, myParams.popSize * myParams.genomeSize);
  // randBankSize = myParams.genomeSize;
  randBankSize = 4 * myParams.popSize * myParams.genomeSize;
  randBank = new float[randBankSize];
  normalBank = new float[randBankSize];

  #pragma omp parallel
  {
    seed_xorshift128plus((uint64_t) random_device()(), (uint64_t) random_device()());
    #pragma omp for schedule(static, myParams.chunkSize)

    for (int i = 0; i < randBankSize; i++)
    {
      randBank[i] = randFloat();
      normalBank[i] = myDist(myEngine);
    }
  }
}

void genetic::mutate(individual &a)
{
  int randOffset = rand() % randBankSize;
  int normalOffset = rand() % randBankSize;

  #pragma omp parallel for schedule(static, myParams.chunkSize)
  for (int i = 0; i < myParams.genomeSize; i++)
  {
    if (randBank[(randOffset + i) % randBankSize] < myParams.mutRate)
    {
      a.genome[i] += normalBank[(normalOffset + i) % randBankSize];
    }
  }
}

void genetic::crossOver(individual &a, individual &b)
{
  int randOffset = rand() % randBankSize;

  #pragma omp parallel for schedule(static, myParams.chunkSize)
  for (int i = 0; i < myParams.genomeSize; i++)
  {
    if (randBank[(randOffset + i) % randBankSize] < myParams.crossRate)
    {
      float temp = b.genome[i];
      b.genome[i] = a.genome[i];
      a.genome[i] = temp;
    }
  }
}

void genetic::getStats()
{
  bestIndIndex = -1;
  totalFitness = 0.0;
  totalScaledFitness = 0.0;
  minFitness = FLT_MAX;
  maxFitness = FLT_MIN;
  for (int i = 0; i < myParams.popSize; i++)
  {
    float f = population[i].fitness;
    totalFitness += f;
    totalScaledFitness += population[i].scaledFitness;

    if (f < minFitness)
    {
      minFitness = f;
    }
    if (f > maxFitness)
    {
      maxFitness = f;
      bestIndIndex = i;
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
    if (randFloat() < population[i].scaledFitness / totalScaledFitness)
    {
      return i;
    }
  }
}

/* only call this if population is already sorted in ascending order! */
void genetic::noRanking()
{
  for (int i = 0; i < myParams.popSize; i++)
  {
    population[i].scaledFitness = population[i].fitness;
  }
}

void genetic::linearRanking()
{
  for (int i = 0; i < myParams.popSize; i++)
  {
    population[i].scaledFitness = i + 1.0f;
  }
}

void genetic::powerRanking()
{
  for (int i = 0; i < myParams.popSize; i++)
  {
    population[i].scaledFitness = pow(i + 1.0f, myParams.alpha);
  }
}

//copy individual b to a
void genetic::copyIndividual(individual &a, individual &b)
{
  a.genome = new float[myParams.genomeSize];
  // copy(b.genome, b.genome + myParams.genomeSize, a.genome);

  #pragma omp parallel for schedule(static, myParams.chunkSize)
  for (int i = 0; i < myParams.genomeSize; i++ )
  {
    a.genome[i] = b.genome[i];
    // float *src = &(b.genome[i]);
    // float *dst = &(a.genome[i]);
    // int length = min(myParams.chunkSize, myParams.genomeSize - i);
    // copy(src, src + length, dst);
  }
  a.fitness = b.fitness;
  a.age = b.age;
}

void genetic::step()
{
  sort(population.begin(), population.end(), compareIndividual);
  powerRanking();
  getStats();

  vector<individual> newPopulation;

  double copyTime = 0.0;
  double crossOverTime = 0.0;
  double mutateTime = 0.0;
  double start;

  for (int i = 0; i < myParams.numToReplace; i += 2)
  {
    int a_i = fitnessSelection();
    int b_i = fitnessSelection();

    individual a, b;

    start = omp_get_wtime();
    copyIndividual(a, population[a_i]);
    copyIndividual(b, population[b_i]);
    copyTime += omp_get_wtime() - start;

    start = omp_get_wtime();
    crossOver(a, b);
    crossOverTime += omp_get_wtime() - start;

    start = omp_get_wtime();
    mutate(a); mutate(b);
    mutateTime += omp_get_wtime() - start;

    newPopulation.push_back(a);
    newPopulation.push_back(b);
  }
  // cout << "copyTime: " << copyTime << endl;
  // cout << "crossOverTime: " << crossOverTime << endl;
  // cout << "mutateTime: " << mutateTime << endl;

  for (int i = 0; i < myParams.numToReplace; i++)
  {
    delete[] population[i].genome;
    population[i] = newPopulation[i];
    population[i].fitness = -1.0;
    population[i].age = 0;
  }
  numGen++;
}

genetic::~genetic()
{
  for (int i = 0; i < myParams.popSize - 1; i++)
  {
    individual a = population[i];
    delete[] a.genome;
  }

  delete[] randBank;
  delete[] normalBank;
}