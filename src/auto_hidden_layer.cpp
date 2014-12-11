#include "auto_hidden_layer.h"
#if HAS_OPENBLAS
#include "cblas.h"
#include "openblas_config.h"
#endif

auto_hidden_layer::auto_hidden_layer(int numInputs, int numHiddenUnits):
  hidden_layer(numInputs, numHiddenUnits)
{
  // hiddenChunkSize = max(numHiddenUnits / 32, 1);
  // inputChunkSize = max(numInputs / 32, 1);
  // hiddenChunkSize = 2;
  // inputChunkSize = 2;

  decode_biases = new float[numInputs];

  //#pragma omp parallel for schedule(static, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    decode_biases[i] = 0.0;
  }

  buffer = new float[numInputs];
}

void auto_hidden_layer::decode(float *input, float *output)
{
#if HAS_OPENBLAS
  resetBuffer();

	cblas_sgemv(CblasRowMajor, CblasTrans, numHiddenUnits, numInputs, 1.0, weights, numInputs, input, 1, 0.0, buffer, 1);

  #pragma omp parallel for schedule(static, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    output[i] = sigmoidTransform(buffer[i] + decode_biases[i]);
  }
#else
  resetBuffer();
  #pragma omp parallel for schedule(static, inputChunkSize)
  for (int j = 0; j < numHiddenUnits; j++)
  {
    for (int i = 0; i < numInputs; i++)
    {
      buffer[i] += weights[j * numInputs + i] * input[j];
    }
  }
  #pragma omp parallel for schedule(static, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    output[i] = sigmoidTransform(buffer[i] + decode_biases[i]);
  }
#endif
}

void auto_hidden_layer::resetBuffer()
{
  //#pragma omp parallel for schedule(static, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    buffer[i] = 0.0;
  }
}

float auto_hidden_layer::auto_squared_loss(float *input, float *output)
{
  float error = 0.0;

  //#pragma omp parallel for schedule(static, inputChunkSize) reduction(+:error)
  for (int i = 0; i < numInputs; i++)
  {
    error += (input[i] - output[i]) * (input[i] - output[i]);
  }

  return 0.5 * error;
}

void auto_hidden_layer::auto_compute_delta_output(float *delta, float *o, float *t)
{
  #pragma omp parallel for schedule(static, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    delta[i] = o[i] * (1.0 - o[i]) * (o[i] - t[i]);
  }
}

void auto_hidden_layer::auto_compute_delta_hidden(float *delta_curr_layer, float *delta_next_layer, float *output_curr_layer)
{
  #pragma omp parallel for schedule(static, hiddenChunkSize)
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

void auto_hidden_layer::auto_updateWeights(float *delta_e, float *o_i, float *delta_d, float *o_e, float learn_rate)
{
  #pragma omp parallel for schedule(static, hiddenChunkSize)
  for (int j = 0; j < numHiddenUnits; j++)
  {
    for (int i = 0; i < numInputs; i++)
    {
      weights[j * numInputs + i] -= learn_rate * o_e[j] * delta_d[i];
    }
  }

  #pragma omp parallel for schedule(static, inputChunkSize)
  for (int i = 0; i < numInputs; i++)
  {
    decode_biases[i] -= learn_rate * delta_d[i];
  }

  #pragma omp parallel for schedule(static, hiddenChunkSize)
  for (int i = 0; i < numHiddenUnits; i++)
  {
    for (int j = 0; j < numInputs; j++)
    {
      weights[i * numInputs + j] -= learn_rate * o_i[j] * delta_e[i];
    }
    biases[i] -= learn_rate * delta_e[i];
  }
}

auto_hidden_layer::~auto_hidden_layer()
{
  delete[] decode_biases;
  delete[] buffer;
}

// cross entrophy loss function
// void auto_hidden_layer::softMaxTransform(float *x)
// {
//   float sum = 0.0;

//   // #pragma omp parallel for schedule(static, numInputs) reduction(+:sum)
//   for (int i = 0; i < numInputs; i++)
//   {
//     sum += exp(x[i]);
//   }

//   #pragma omp parallel for schedule(static, numInputs)
//   for (int i = 0; i < numInputs; i++)
//   {
//     x[i] = exp(x[i]) / sum;
//   }
// }

// void auto_hidden_layer::cross_decode(float *input, float *output)
// {
//   #pragma omp parallel for schedule(static, inputChunkSize)
//   for (int i = 0; i < numInputs; i++)
//   {
//     float sum = 0.0;
//     for (int j = 0; j < numHiddenUnits; j++)
//     {
//       sum += weights[j * numInputs + i] * input[j];
//     }
//     output[i] = sum + decode_biases[i];
//   }
//   softMaxTransform(output);
// }

// void auto_hidden_layer::cross_compute_delta_output(float *delta, float *o, float *t)
// {
//   #pragma omp parallel for schedule(static, inputChunkSize)
//   for (int i = 0; i < numInputs; i++)
//   {
//     delta[i] = (o[i] - t[i]);
//   }
// }

// float auto_hidden_layer::cross_entropy_loss(float *input, float *output)
// {
//   float error = 0.0;
//   #pragma omp parallel for schedule(static, inputChunkSize) reduction(+:error)
//   for (int i = 0; i < numInputs; i++)
//   {
//     // float o = ((output[i] < .5) ? max((float)1e-6, output[i]) : min(1 - (float)1e-6, output[i]));
//     error += input[i] * log(max(output[i], 1e-6f)) + (1 - input[i]) * log(max(1 - output[i], 1e-6f));
//   }
//   return -1 * error;
// }
