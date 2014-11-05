#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "hidden_layer.h"

class output_layer: public hidden_layer
{
public:
	using hidden_layer::hidden_layer;
  void encode(float *input, float *output);
	void compute_delta_output(float *delta, float *o, int t);
  void softMaxTransform(float *x);
	float cross_entropy_loss(float *output, int t);
};

#endif
