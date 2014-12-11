#include "neural_network.h"

neural_network::neural_network(int numInput, int numHidden, int numOutput, float learn_rate):
  learn_rate(learn_rate)
{
  cout << "created a squared loss neural network with " << numInput << " input, " << numHidden << " hidden, " << numOutput << " output units and " << learn_rate << " learn rate" << endl;

  h = new hidden_layer(numInput, numHidden);
  o = new hidden_layer(numHidden, numOutput);

  o_j = new float[h->getNumHiddenUnits()];
  o_k = new float[o->getNumHiddenUnits()];
  delta_k = new float[o->getNumHiddenUnits()];
  delta_j = new float[h->getNumHiddenUnits()];
}

void neural_network::train(float **trainingImages, vector<int> &trainLabels, int numOuterIter, int numTrainingImages)
{
  cout << "cycling through " << numTrainingImages << " training images for " << numOuterIter << " outer iterations" << endl;
  float sum_squared_error = 0.0;
  for (int j = 0; j < numTrainingImages; j++)
  {
  	// cout << "image " << j << endl;
    float *o_i = trainingImages[j];
    // for (int i = 0; i < h->getNumInputUnits(); i++)
    // {
    // 	float x = o_i[i];
    // }
    // cout << j << " image ok" << endl;
    h->encode(o_i, o_j);
    // cout << j << " encode ok" << endl;
    o->encode(o_j, o_k);
    // cout << j << " output encode ok" << endl;
    sum_squared_error += o->squared_loss(o_k, trainLabels[j]);
  }
  cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

  double start = omp_get_wtime();
  for (int i = 0; i < numOuterIter; i++ )
  {
    float sum_squared_error = 0.0;
    for (int j = 0; j < numTrainingImages; j++)
    {
      float *o_i = trainingImages[j];
      sum_squared_error += backprop(o_i, trainLabels[j]);
    }
    cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
  }
}

void neural_network::test(float **testingImages, vector<int> &testLabels, int numTestingImages)
{
  cout << "evaluating network on " << numTestingImages << " test digits" << endl;
  float correct = 0;
  for (int i = 0; i < numTestingImages; i++)
  {
    float *o_i = testingImages[i];
    getInput(o_i);
    int predict_value = predict(o_i);
    if (predict_value == testLabels[i])
    {
      correct += 1;
    }
  }
  cout << "accuracy rate: " << correct / numTestingImages << endl;
}

float neural_network::backprop(float *o_i, int t)
{
  h->encode(o_i, o_j);
  o->encode(o_j, o_k);

  o->compute_delta_output(delta_k, o_k, t);
  h->compute_delta_hidden(delta_j, delta_k, o_j, o);

  o->updateWeights(delta_k, o_j, learn_rate);
  h->updateWeights(delta_j, o_i, learn_rate);
  return o->squared_loss(o_k, t);
}

int neural_network::predict(float *o_i)
{
  h->encode(o_i, o_j);
  o->encode(o_j, o_k);

  int bestIndex = -1;
  float bestValue = -1;
  for (int i = 0; i < (int) o->getNumHiddenUnits(); i++)
  {
    // cout << o_k[i] << " ";
    if (o_k[i] > bestValue)
    {
      bestValue = o_k[i];
      bestIndex = i;
    }
  }
  // cout << endl;
  return bestIndex;
}

neural_network::~neural_network()
{
  delete o;
  delete h;
  delete[] o_j;
  delete[] o_k;
  delete[] delta_k;
  delete[] delta_j;
}
