#include "output_layer.h"

/*
void softMaxTransform(float *x)
{
  float sum = 0.0;
  for (int i = 0; i < numHiddenUnits; i++)
  {
    sum += x[i];
  }

  for (int i = 0; i < numHiddenUnits; i++)
  {
    x[i] = 1 / (1 + exp(-x/sum));
  }
}
*/

void output_layer::softMaxTransform(float *x)
{
  float sum = 0.0;

  #pragma omp parallel for schedule(dynamic) reduction(+:sum)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    sum += exp(x[i]);
  }

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    x[i] = exp(x[i]) / sum;
  }
}


void output_layer::encode(float *input, float *output)
{
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numInputs; j++)
    {
      sum += weights[i * numInputs + j] * input[j];
    }
    output[i] = sum + biases[i];
  }
  softMaxTransform(output);
}


void output_layer::compute_delta_output(float *delta, float *o, int t)
{
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    __t[i] = 0.0;
  }
  __t[t] = 1.0;

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    delta[i] = (o[i] - __t[i]);
  }
}

float output_layer::cross_entropy_loss(float *output, int t)
{
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    __t[i] = 0.0;
  }
  __t[t] = 1.0;

  float error = 0.0;

  #pragma omp parallel for schedule(dynamic) reduction(+:error)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float o = ((output[i] < .5) ? max((float)1e-6, output[i]) : min(1 - (float)1e-6, output[i]));
    error += __t[i] * log(o) + (1 - __t[i]) * log(1 - o);
  }

  return -1 * error;
}
