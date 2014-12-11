#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "hidden_layer.h"
#include "output_layer.h"

using namespace std;

class neural_network
{
protected:
  hidden_layer *h, *o;
  float *o_j, *o_k, *delta_k, *delta_j;
  float learn_rate, prevTrainError;
  // int numHiddenLayers;
  // vector<*hidden_layer> myLayers;

public:
  neural_network(int numInput, int numHidden, int numOutput, float learn_rate);
  virtual void train(float **trainingImages, vector<int> &trainLabels, int numOuterIter, int numTrainingImages);
  void test(float **testingImages, vector<int> &testLabels, int numTestingImages);
  virtual void getInput(float *&o_i) {};
  virtual float backprop(float *o_i, int t);
  virtual int predict(float *o_i);
  virtual ~neural_network();
};

#endif
