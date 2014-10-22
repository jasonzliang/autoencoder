#include "hidden_layer.h"

float RandomNumber(float Min, float Max)
{
  return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

void sigmoidTransform(float *x)
{
  for (int i = 0; i < numWeights; i++)
  {
    x[i] = 1 / (1 + exp(-x));
  }
}

hidden_layer::hidden_layer(int numInputs, int numHiddenUnits):
  numInputs(numInputs),
  numHiddenUnits(numHiddenUnits)
{
  weightRange = 4. * sqrt(6. / (numInputs + numHiddenUnits));
  init();
}

hidden_layer::hidden_layer(int numInputs, int numHiddenUnits, float weightRange):
  numInputs(numInputs),
  numHiddenUnits(numHiddenUnits),
  weightRange(weightRange)
{
  init();
}

void hidden_layer::init()
{
  numWeights = numInputs * numHiddenUnits;
  weights = new float[numWeights];

  for (int i = 0; i < numWeights; i++)
  {
    weights[i] = RandomNumber(-weightRange, weightRange);
  }
}

void hidden_layer::encode(float *input, float *output)
{
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numInputs; j++)
    {
      sum += weights[i * numInputs + j] * input[j];
    }
    output[i] = sum;
  }
  sigmoidTransform(output);
}

void hidden_layer::decode(float *input, float *output)
{
  for (int i = 0; i < numInputs; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numHiddenUnits; j++)
    {
      sum += weights[i + numHiddenUnits * j] * input[j];
    }
    output[i] = sum;
  }
  sigmoidTransform(output);
}

float hidden_layer::loss_function(float *input)
{
  float hiddenValues[numHiddenUnits];
  float output[numInputs];
  encode(input, &hiddenValues);
  decode(&hiddenValues, &output);

  float error = 0.0;
  for (int i = 0; i < numInputs; i++)
  {
  	error += pow(input[i] - output[i], 2);
  }
  return 0.5*error;
}

hidden_layer::~hidden_layer()
{
  delete[] weights;
}