#include "hidden_layer.h"

float RandomNumber(float Min, float Max)
{
  return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

void sigmoidTransform(float *x, int numHiddenUnits)
{
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    x[i] = 1 / (1 + exp(-1 * x[i]));
  }
}

void softMaxTransform(float *x, int numHiddenUnits)
{
  float sum = 0.0;

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    sum += x[i];
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    x[i] = 1 / (1 + exp(-1 * x[i] / sum));
  }
}

hidden_layer::hidden_layer(int numInputs, int numHiddenUnits):
  numInputs(numInputs),
  numHiddenUnits(numHiddenUnits)
{
  weightRange = 1.0 / sqrt(numInputs);
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

  #pragma omp parallel for schedule(dynamic, numHiddenUnits)
  for (int i = 0; i < numWeights; i++)
  {
    weights[i] = RandomNumber(-weightRange, weightRange);
  }

  biases = new float[numHiddenUnits];

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    biases[i] = 0.0;
  }

  __t = new float[numHiddenUnits];
}

void hidden_layer::encode(float *input, float *output)
{
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numInputs; j++)
    {
      sum += weights[i * numInputs + j] * input[j];
    }
    output[i] = sum + biases[i];
  }
  sigmoidTransform(output, numHiddenUnits);
}

void hidden_layer::decode(float *input, float *output)
{
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numInputs; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numHiddenUnits; j++)
    {
      sum += weights[i + numHiddenUnits * j] * input[j];
    }
    output[i] = sum;
  }
  sigmoidTransform(output, numHiddenUnits);
}

float hidden_layer::autoencoder_squared_loss(float *input)
{
  float hiddenValues[numHiddenUnits];
  float output[numInputs];
  encode(input, hiddenValues);
  decode(hiddenValues, output);

  float error = 0.0;

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numInputs; i++)
  {
    error += pow(input[i] - output[i], 2);
  }
  return 0.5 * error;
}

float hidden_layer::squared_loss(float *output, int t)
{
  float error = 0.0;
  
  for (int i = 0; i < numHiddenUnits; i++)
  {
    error += pow(output[i] - __t[i], 2);
  }
  return 0.5 * error;
}

void hidden_layer::compute_delta_output(float *delta, float *o, int t)
{
  for (int i = 0; i < numHiddenUnits; i++)
  {
    __t[i] = 0.0;
  }
  __t[t] = 1.0;

  for (int i = 0; i < numHiddenUnits; i++)
  {
    delta[i] = o[i] * (1.0 - o[i]) * (o[i] - __t[i]);
  }
}

void hidden_layer::compute_delta_hidden(float *delta_curr_layer, float *delta_next_layer, float *output_curr_layer, hidden_layer *next_layer)
{
  //i is actually j
  //j is actually k
  float *output_layer_weights = next_layer->getWeights();
  int numHidUnits_nextLayer = next_layer->getNumHiddenUnits();

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;

    for (int j = 0; j < numHidUnits_nextLayer; j++)
    {
      //we are iterating through ith column of next layer's weight matrix
      sum += delta_next_layer[j] * output_layer_weights[j * numHiddenUnits + i];
    }
    delta_curr_layer[i] = output_curr_layer[i] * (1 - output_curr_layer[i]) * sum;
  }
}

void hidden_layer::updateWeights(float *delta_curr_layer, float *output_prev_layer, float learn_rate)
{
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    for (int j = 0; j < numInputs; j++)
    {
      weights[i * numInputs + j] -= learn_rate * output_prev_layer[j] * delta_curr_layer[i];
    }
  }

  for (int i = 0 ; i < numHiddenUnits; i++)
  {
    biases[i] += learn_rate * delta_curr_layer[i];
  }
}

hidden_layer::~hidden_layer()
{
  delete[] weights;
  delete[] biases;
  delete[] __t;
}