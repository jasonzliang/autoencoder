#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "hidden_layer.h"
#include "output_layer.h"

using namespace std;

class neural_network
{
private:
  //hidden_layer *h, *o;
	hidden_layer *h;
	output_layer *o;
  float *o_j, *o_k, *delta_k, *delta_j;
	bool cross_entropy;

  float learn_rate;
  // int numHiddenLayers;
  // vector<*hidden_layer> myLayers;

public:
  neural_network(int numInput, int numHidden, int numOutput, float learn_rate, bool cross_entropy_flag);
  float backprop(float *o_i, int t);
  int predict(float *o_i);
  ~neural_network();
};

#endif
