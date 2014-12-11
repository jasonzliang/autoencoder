#include "hidden_layer.h"

#ifndef AUTO_HIDDEN_LAYER
#define AUTO_HIDDEN_LAYER

using namespace std;

class auto_hidden_layer: public hidden_layer
{
private:
  float *decode_biases, *buffer;

public:
  auto_hidden_layer(int numInputs, int numHiddenUnits);
  ~auto_hidden_layer();

  void decode(float *input, float *output);
  void resetBuffer();

  float auto_squared_loss(float *input, float *output);
  void auto_compute_delta_output(float *delta, float *o, float *t);
  void auto_compute_delta_hidden(float *delta_curr_layer, float *delta_next_layer, float *output_curr_layer);
  void auto_updateWeights(float *delta_e, float *o_i, float *delta_d, float *o_e, float learn_rate);

  inline float* getDecodeBiases() {
    return decode_biases;
  }

  inline void setDecodeBiases(float *newBiases) {
    decode_biases = newBiases;
  }
  //cross entrophy-loss functions
  // void softMaxTransform(float *x);
  // void cross_decode(float *input, float *output);
  // void cross_compute_delta_output(float *delta, float *o, float *t);
  // float cross_entropy_loss(float *input, float *output);
};

#endif