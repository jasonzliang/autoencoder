#include "auto_hidden_layer.h"

auto_hidden_layer::auto_hidden_layer(int numInputs, int numHiddenUnits):
  hidden_layer(numInputs, numHiddenUnits)
{
	hiddenChunkSize = max(numHiddenUnits/32, 1);
	inputChunkSize = max(numInputs/32, 1);
	// hiddenChunkSize = 64;
	// inputChunkSize = 64;

  decode_biases = new float[numInputs];

  #pragma omp parallel for schedule(dynamic, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    decode_biases[i] = 0.0;
  }
}

void auto_hidden_layer::decode(float *input, float *output)
{
  #pragma omp parallel for schedule(dynamic, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numHiddenUnits; j++)
    {
      sum += weights[j * numHiddenUnits + i] * input[j];
    }
    output[i] = sigmoidTransform(sum + decode_biases[i]);
  }
}

float auto_hidden_layer::squared_loss(float *input, float *output)
{
  float error = 0.0;

  #pragma omp parallel for schedule(dynamic, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    error += (input[i] - output[i])*(input[i] - output[i]);
  }

  return 0.5 * error;
}

void auto_hidden_layer::compute_delta_output(float *delta, float *o, float *t)
{
  #pragma omp parallel for schedule(dynamic, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    delta[i] = o[i] * (1.0 - o[i]) * (o[i] - t[i]);
  }
}

void auto_hidden_layer::compute_delta_hidden(float *delta_curr_layer, float *delta_next_layer, float *output_curr_layer)
{
  #pragma omp parallel for schedule(dynamic, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;

    for (int j = 0; j < numInputs; j++)
    {
      //we are iterating through ith column of next layer's weight matrix
      sum += delta_next_layer[j] * weights[i * numInputs + j];
    }
    delta_curr_layer[i] = output_curr_layer[i] * (1 - output_curr_layer[i]) * sum;
  }
}

void auto_hidden_layer::updateWeights(float *delta_e, float *o_i, float *delta_d, float *o_e, float learn_rate)
{
  #pragma omp parallel for schedule(dynamic, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    for (int j = 0; j < numHiddenUnits; j++)
    {
      weights[j * numHiddenUnits + i] -= learn_rate * o_e[j] * delta_d[i];
    }
    decode_biases[i] += learn_rate * delta_d[i];
  }

  #pragma omp parallel for schedule(dynamic, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    for (int j = 0; j < numInputs; j++)
    {
      weights[i * numInputs + j] -= learn_rate * o_i[j] * delta_e[i];
    }
    biases[i] += learn_rate * delta_e[i];
  }
}

auto_hidden_layer::~auto_hidden_layer()
{
  delete[] decode_biases;
}