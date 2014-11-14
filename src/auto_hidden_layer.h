#include "hidden_layer.h"

#ifndef AUTO_HIDDEN_LAYER
#define AUTO_HIDDEN_LAYER

using namespace std;

class auto_hidden_layer: public hidden_layer
{
private:
	float *decode_biases;
	int hiddenChunkSize, inputChunkSize;

public:
	auto_hidden_layer(int numInputs, int numHiddenUnits);
	~auto_hidden_layer();

  void decode(float *input, float *output);
  
  float squared_loss(float *input, float *output);
  void compute_delta_output(float *delta, float *o, float *t);
  void compute_delta_hidden(float *delta_curr_layer, float *delta_next_layer, float *output_curr_layer);
  void updateWeights(float *delta_e, float *o_i, float *delta_d, float *o_e, float learn_rate);
};

#endif