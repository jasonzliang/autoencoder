#ifndef HIDDEN_LAYER
#define HIDDEN_LAYER

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class hidden_layer
{
private:
  int numInputs, numHiddenUnits, numWeights;
  float weightRange;
  float *weights, *biases, *__t;

public:
  hidden_layer(int numInputs, int numHiddenUnits);
  hidden_layer(int numInputs, int numHiddenUnits, float weightRange);
  virtual ~hidden_layer();
  void init();
  virtual void encode(float *input, float *output);
  void decode(float *input, float *output);
  float autoencoder_squared_loss(float *input);
  float squared_loss(float *output, int t);


  void compute_delta_output(float *delta, float *o, int t);
  void compute_delta_hidden(float *delta_curr_layer, float *delta_next_layer, float *output_curr_layer, hidden_layer *next_layer);
  void updateWeights(float *delta_curr_layer, float *output_prev_layer, float learn_rate);

  inline int getNumHiddenUnits()
  {
    return numHiddenUnits;
  }
  inline int getNumInputUnits()
  {
    return numInputs;
  }
  inline int getNumWeights()
  {
    return numWeights;
  }
  inline float *getWeights()
  {
  	return weights;
  }
};

#endif