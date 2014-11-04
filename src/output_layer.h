#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "hidden_layer.h"

class output_layer: public hidden_layer
{
public:
  void encode(float *input, float *output);
};

#endif