#include "hidden_layer.h"

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
  hiddenChunkSize = max(numHiddenUnits / 32, 1);
  inputChunkSize = max(numInputs / 32, 1);

  numWeights = numInputs * numHiddenUnits;
  weights = new float[numWeights];

  #pragma omp parallel for schedule(dynamic, numHiddenUnits)
  for (int i = 0; i < numWeights; i++)
  {
    weights[i] = RandomNumber(-weightRange, weightRange);
  }

  biases = new float[numHiddenUnits];

  //#pragma omp parallel for schedule(dynamic, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    biases[i] = 0.0;
  }

  __t = new float[numHiddenUnits];
}

void hidden_layer::printWeights(int n)
{
  n = min(numHiddenUnits, n);
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < numInputs; j++)
    {
      cout << weights[i * numInputs + j] << " ";
    }
    cout << endl;
  }
}

void hidden_layer::sigmoidTransform(float *x)
{
  #pragma omp parallel for schedule(dynamic, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    x[i] = 1 / (1 + exp(-1 * x[i]));
  }
}

void hidden_layer::encode(float *input, float *output)
{
  #pragma omp parallel for schedule(dynamic, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numInputs; j++)
    {
      sum += weights[i * numInputs + j] * input[j];
    }
    output[i] = sigmoidTransform(sum + biases[i]);
  }
}

float hidden_layer::squared_loss(float *output, int t)
{
  for (int i = 0; i < numHiddenUnits; i++)
  {
    __t[i] = 0.0;
  }
  __t[t] = 1.0;

  float error = 0.0;
  for (int i = 0; i < numHiddenUnits; i++)
  {
    error += (output[i] - __t[i]) * (output[i] - __t[i]);
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

  #pragma omp parallel for schedule(dynamic, hiddenChunkSize)
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
  #pragma omp parallel for schedule(dynamic, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    //#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < numInputs; j++)
    {
      weights[i * numInputs + j] -= learn_rate * output_prev_layer[j] * delta_curr_layer[i];
    }
    biases[i] += learn_rate * delta_curr_layer[i];
  }

}

hidden_layer::~hidden_layer()
{
  delete[] weights;
  delete[] biases;
  delete[] __t;
}
