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
  float* weights;

public:
  hidden_layer(int numInputs, int numHiddenUnits);
  hidden_layer(int numInputs, int numHiddenUnits, float weightRange);
  ~hidden_layer();
  void init();
	void encode(float *input, float *output);
	void decode(float *input, float *output);
	float loss_function(float *input);
};

#endif