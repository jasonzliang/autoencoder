#ifndef NEURAL_NETWORK_CROSS
#define NEURAL_NETWORK_CROSS

#include "hidden_layer.h"
#include "output_layer.h"

using namespace std;

class neural_network_cross
{
private:
	hidden_layer *h;
	output_layer *o;
  float *o_j, *o_k, *delta_k, *delta_j;

  float learn_rate;
  // int numHiddenLayers;
  // vector<*hidden_layer> myLayers;

public:
  neural_network_cross(int numInput, int numHidden, int numOutput, float learn_rate);
  float backprop(float *o_i, int t);
  int predict(float *o_i);
  ~neural_network_cross();
};

#endif
