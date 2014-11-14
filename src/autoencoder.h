#ifndef AUTOENCODER
#define AUTOENCODER

#include "auto_hidden_layer.h"
#include "neural_network.h"

using namespace std;

class autoencoder: public neural_network
{
private:
	int numPreTrainLayers;
  vector<auto_hidden_layer *> preTrainLayers;
  vector<float *> preTrainLayersOutputs;

public:
  autoencoder(vector<int> preTrainLayerWidths, int numInput, int numHidden, int numOutput, float learn_rate);
  ~autoencoder();

  void preTrain(float **trainingImages, int numTrainingImages, int numOuterIter);
};

#endif