#include "output_layer.h"

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

void output_layer::encode(float *input, float *output)
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
  softMaxTransform(output);
}
