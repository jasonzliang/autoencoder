#include "neural_network.h"

neural_network::neural_network(int numInput, int numHidden, int numOutput, float learn_rate, bool cross_entropy):
	cross_entropy(cross_entropy),
  learn_rate(learn_rate)
{
  cout << "created a neural network with " << numInput << " input, " << numHidden << " hidden, " << numOutput << " output units and " << learn_rate << " learn rate" << endl;
  h = new hidden_layer(numInput, numHidden);
	if(cross_entropy) oc = new output_layer(numHidden, numOutput);
	else o = new hidden_layer(numHidden, numOutput);

  o_j = new float[h->getNumHiddenUnits()];
  o_k = new float[o->getNumHiddenUnits()];
  delta_k = new float[o->getNumHiddenUnits()];
  delta_j = new float[h->getNumHiddenUnits()];
}

float neural_network::backprop(float *o_i, int t)
{
	if(cross_entropy)
	{
		h->encode(o_i, o_j);
		oc->encode(o_j, o_k);
		

		oc->compute_delta_output(delta_k, o_k, t);
		h->compute_delta_hidden(delta_j, delta_k, o_j, o);

		oc->updateWeights(delta_k, o_j, learn_rate);
		h->updateWeights(delta_j, o_i, learn_rate);

		return oc->cross_entropy_loss(o_k, t);
	}
	else{
		h->encode(o_i, o_j);
		o->encode(o_j, o_k);
		

		o->compute_delta_output(delta_k, o_k, t);
		h->compute_delta_hidden(delta_j, delta_k, o_j, o);

		o->updateWeights(delta_k, o_j, learn_rate);
		h->updateWeights(delta_j, o_i, learn_rate);
		return o->squared_loss(o_k, t);
	}
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
  	if (o_k[i] > bestValue) {
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
