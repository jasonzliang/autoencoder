#include "autoencoder.h"

inline float uniformRandom()
{
  return ( (float)(rand()) + 1. ) / ( (float)(RAND_MAX) + 1. );
}

inline float normalRandom()
{
  float u1 = uniformRandom();
  float u2 = uniformRandom();
  return cos(8.*atan(1.) * u2) * sqrt(-2.*log(u1));
}

autoencoder::autoencoder(vector<int> preTrainLayerWidths, vector<float> preTrainLayersLearnRates, vector<int> preTrainLayersOuterIter, vector<float> preTrainLayersNoiseLevels, int numInput, int numHidden, int numOutput, float learn_rate):
  neural_network(numInput, numHidden, numOutput, learn_rate),
  preTrainLayersLearnRates(preTrainLayersLearnRates),
  preTrainLayersOuterIter(preTrainLayersOuterIter),
  preTrainLayersNoiseLevels(preTrainLayersNoiseLevels)
	output_layer *output = new output_layer(preTrainLayerWidths[numPreTrainLayers],10);


{
  numPreTrainLayers = (int) preTrainLayerWidths.size();
  cout << "created an stacked denoising autoencoder with " << numPreTrainLayers << " pretrained layers" << endl;
  for (int i = 0; i < numPreTrainLayers; i++)
  {
    int auto_num_input = preTrainLayerWidths[i];
    int auto_num_output;
    if (i == numPreTrainLayers - 1)
    {
      auto_num_output = numInput;
    }
    else
    {
      auto_num_output = preTrainLayerWidths[i + 1];
    }

    auto_hidden_layer *x = new auto_hidden_layer(auto_num_input, auto_num_output);
    preTrainLayers.push_back(x);
    float *output = new float[auto_num_output];
    preTrainLayersOutputs.push_back(output);

    cout << "layer " << i + 1 << ": number of outer iterations " << preTrainLayersOuterIter[i] << ", learning rate " << preTrainLayersLearnRates[i] << ", number of hidden units " << auto_num_output << ", noise level " << preTrainLayersNoiseLevels[i] << endl;
  }

}

void autoencoder::corrupt_masking(float *input, float *corrupted_input, float fraction, int length)
{
  // #pragma omp parallel for schedule(static, 2)
  for (int i = 0; i < length; i++)
  {
    if (uniformRandom() < fraction)
    {
      corrupted_input[i] = 0.0;
    }
    else
    {
      corrupted_input[i] = input[i];
    }
  }
}

void autoencoder::corrupt_gaussian(float *input, float *corrupted_input, float sigma, int length)
{
  // #pragma omp parallel for schedule(static, 50)
  for (int i = 0; i < length; i++)
  {
    corrupted_input[i] = input[i] + normalRandom() * sigma;
  }
}

void autoencoder::getInputK(float *&o_i, int k)
{
  for (int i = 0; i < k; i++)
  {
    preTrainLayers[i]->encode(o_i, preTrainLayersOutputs[i]);
    o_i = preTrainLayersOutputs[i];
  }
}

void autoencoder::getInput(float *&o_i)
{
  for (int i = 0; i < numPreTrainLayers; i++)
  {
    preTrainLayers[i]->encode(o_i, preTrainLayersOutputs[i]);
    o_i = preTrainLayersOutputs[i];
  }
}

void autoencoder::preTrain(float **trainingImages, int numTrainingImages)
{
  for (int k = 0; k < numPreTrainLayers; k++)
  {

    cout << "pretraining layer #" << k + 1 << ", cycling through " << numTrainingImages << " training images for " << preTrainLayersOuterIter[k] << " outer iterations" << endl;

    auto_hidden_layer *a = preTrainLayers[k];
    float *corrupted_o_i = new float[a->getNumInputUnits()];
    float *o_e = new float[a->getNumHiddenUnits()];
    float *delta_e = new float[a->getNumHiddenUnits()];
    float *o_d = new float[a->getNumInputUnits()];
    float *delta_d = new float[a->getNumInputUnits()];

    float sum_squared_error = 0.0;
    for (int j = 0; j < numTrainingImages; j++)
    {
      float *o_i = trainingImages[j];
      getInputK(o_i, k);

      a->encode(o_i, o_e);
      a->decode(o_e, o_d);

      sum_squared_error += a->auto_squared_loss(o_i, o_d);
    }
    cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;
    prevTrainError = sum_squared_error;
    float initLearningRate = preTrainLayersLearnRates[k];
    float thres = 0.02;

    double start = omp_get_wtime();
    for (int i = 0; i < preTrainLayersOuterIter[k]; i++ )
    {
      // float total_encode_time = 0;
      // float total_decode_time = 0;
      // float total_compute_delta_output_time = 0;
      // float total_compute_delta_hidden_time = 0;
      // float total_updateWeights_time = 0;
      // float total_squared_loss_time = 0;
      // float t;

      float sum_squared_error = 0.0;
      for (int j = 0; j < numTrainingImages; j++)
      {
        float *o_i = trainingImages[j];
        getInputK(o_i, k);
        corrupt_masking(o_i, corrupted_o_i, preTrainLayersNoiseLevels[k], a->getNumInputUnits());

        // t = omp_get_wtime();
        a->encode(corrupted_o_i, o_e);
        // total_encode_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->decode(o_e, o_d);
        // total_decode_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->auto_compute_delta_output(delta_d, o_d, o_i);
        // total_compute_delta_output_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->auto_compute_delta_hidden(delta_e, delta_d, o_e);
        // total_compute_delta_hidden_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->auto_updateWeights(delta_e, corrupted_o_i, delta_d, o_e, preTrainLayersLearnRates[k]);
        // total_updateWeights_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        sum_squared_error += a->auto_squared_loss(o_i, o_d);
        // total_squared_loss_time += omp_get_wtime() - t;

      }
      cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
      // cout << "total_encode_time: " << total_encode_time << endl;
      // cout << "total_decode_time: " << total_decode_time << endl;
      // cout << "total_compute_delta_output_time: " << total_compute_delta_output_time << endl;
      // cout << "total_compute_delta_hidden_time: " << total_compute_delta_hidden_time << endl;
      // cout << "total_updateWeights_time: " << total_updateWeights_time << endl;
      // cout << "total_squared_loss_time: " << total_squared_loss_time << endl;
      // cout << endl;
      if (sum_squared_error > (1-thres)*prevTrainError) {
        preTrainLayersLearnRates[k] /= 2;
        thres /= 2;
        cout << "error converging, using new learning rate " << preTrainLayersLearnRates[k] << endl;
      }
      prevTrainError = sum_squared_error;
    }

    delete[] corrupted_o_i;
    delete[] o_e;
    delete[] o_d;
    delete[] delta_e;
    delete[] delta_d;
  }
}

void autoencoder::visualizeWeights(int layer, int n)
{
  cout << "printing weights for first " << n << " hidden units of layer " << layer + 1 << endl;
  auto_hidden_layer *myLayer = preTrainLayers[layer];
  myLayer->printWeights(n);
}

void autoencoder::reconstructImage(float **testingImages, int layer, int n)
{
  cout << "printing " << n << " reconstructed test images for " << layer + 1 << endl;
  auto_hidden_layer *myLayer = preTrainLayers[layer];
  float *myLayerOutput = preTrainLayersOutputs[layer];
  float *reconstructed_output = new float[784];
  float *corrupted_input = new float[784];
  for (int i = 0; i < n; i++)
  {
    float *image = testingImages[i];
    corrupt_masking(image, corrupted_input, 0.25, 784);
    myLayer->encode(corrupted_input, myLayerOutput);
    myLayer->decode(myLayerOutput, reconstructed_output);
    for (int j = 0; j < 784; j++)
    {
      cout << corrupted_input[j] << " ";
    }
    cout << endl;
    for (int j = 0; j < 784; j++)
    {
      cout << reconstructed_output[j] << " ";
    }
    cout << endl;
  }
  delete[] reconstructed_output;
  delete[] corrupted_input;
}

autoencoder::~autoencoder()
{
  while (!preTrainLayers.empty())
  {
    delete preTrainLayers.back();
    preTrainLayers.pop_back();
  }
  while (!preTrainLayersOutputs.empty())
  {
    delete[] preTrainLayersOutputs.back();
    preTrainLayersOutputs.pop_back();
  }
}

void autoencoder::fineTune(float **trainingImages, int numTrainingImages, vector<int> &trainLabels){
	cout << "STARTING THE FINE TUNING STEP" << endl;

	int numOuterIter = 5;

	cout << "cycling through " << numTrainingImages << " training images for " << numOuterIter << " outer iterations" << endl;
	float sum_squared_error = 0.0;

	// Forward activiate the network to compute the initial output error
	/*
	for (int j = 0; j < numTrainingImages; j++)
	{
		float *o_i = trainingImages[j];
		getInput(o_i);

		h->encode(o_i, o_j);
		o->encode(o_j, o_k);
		sum_squared_error += o->squared_loss(o_k, trainLabels[j]);
	}
	*/
	sum_squared_error = 1234123;
	cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

	double start = omp_get_wtime();
	for (int i = 0; i < numOuterIter; i++ ) // TODO should make this an input probably
	{
		float sum_squared_error = 0.0;

		int p = numPreTrainLayers;
		for (int j = 0; j < numTrainingImages; j++)
		{
			// compute all forward activations
			float *o_i = trainingImages[j];
			float **activations;
			float **deltas;
			activations =  new float*[p+2];
			deltas =  new float*[p+2];
			for(int k=0; k<p;k++){
				auto_hidden_layer *a = preTrainLayers[k];
				activations[k] = new float[a->getNumHiddenUnits()];
				deltas[k] = new float[a->getNumHiddenUnits()];
				if(k==0){
					a->encode(o_i, activations[k]);
				}
				else{
					a->encode(activations[k-1], activations[k]);
				}
			}
			activations[p] = new float[h->getNumHiddenUnits()];
			h->encode(activations[p-1], activations[p]);
			deltas[p] = new float[h->getNumHiddenUnits()];
			activations[p+1] = new float[o->getNumHiddenUnits()];
			o->encode(activations[p], activations[p+1]);
			deltas[p+1] = new float[o->getNumHiddenUnits()];
			// all activations computed
			
			// Now compute the deltas for the hidden/output layers (not the autoencoder layers)
			o->compute_delta_output(deltas[p+1], activations[p+1], trainLabels[j]);
			h->compute_delta_hidden(deltas[p], deltas[p+1], activations[p], o);

  		o->updateWeights(deltas[p+1], activations[p], learn_rate);
  		h->updateWeights(deltas[p], activations[p-1], learn_rate);

			// get layers in order starting from the last one and we'll update in that order
			for (int k = numPreTrainLayers-1; k > 1; k--) // TODO should this really only go until 1?
			{
				auto_hidden_layer *a = preTrainLayers[k];
				// we use the compute delta hidden from the neural net as opposed to the
				// autoencoder since we don't want to compare the original with encoded/decoded images.
				if(k == p-1){
					a->compute_delta_hidden(deltas[k], deltas[k+1], activations[k], h);
				}
				else{
					a->compute_delta_hidden(deltas[k], deltas[k+1], activations[k], preTrainLayers[k+1]);
				}
				a->updateWeights(deltas[k], activations[k-1], preTrainLayersLearnRates[k]);
				// Need to add fine tuning for the weights from the input to the first activation layer here.
			}
			
			// after updating all the weights we compute the error again
			for(int k=0; k<p;k++){
				auto_hidden_layer *a = preTrainLayers[k];
				if(k==0){
					a->encode(o_i, activations[k]);
				}
				else{
					a->encode(activations[k-1], activations[k]);
				}
			}
			h->encode(activations[p-1], activations[p]);
			o->encode(activations[p], activations[p+1]);

			sum_squared_error += o->squared_loss(activations[p+1], trainLabels[j]);

			// Free the memory that we allocated
			for(int k=0; k<p;k++){
				delete[] activations[k];
				delete[] deltas[k];
			}
			delete activations;
			delete deltas;
		}
		cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
	}


	return;
}


void autoencoder::fineTuneNoHidden(float **trainingImages, int numTrainingImages, vector<int> &trainLabels){
	cout << "STARTING THE FINE TUNING STEP" << endl;

	int numOuterIter = 5;

	cout << "cycling through " << numTrainingImages << " training images for " << numOuterIter << " outer iterations" << endl;
	float sum_squared_error = 0.0;

	// Forward activiate the network to compute the initial output error
	/*
	for (int j = 0; j < numTrainingImages; j++)
	{
		float *o_i = trainingImages[j];
		getInput(o_i);

		h->encode(o_i, o_j);
		o->encode(o_j, o_k);
		sum_squared_error += o->squared_loss(o_k, trainLabels[j]);
	}
	*/
	sum_squared_error = 1234123;
	cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

	double start = omp_get_wtime();
	for (int i = 0; i < numOuterIter; i++ ) // TODO should make this an input probably
	{
		float sum_squared_error = 0.0;

		int p = numPreTrainLayers;
		for (int j = 0; j < numTrainingImages; j++)
		{
			// compute all forward activations
			float *o_i = trainingImages[j];
			float **activations;
			float **deltas;
			activations =  new float*[p+1];
			deltas =  new float*[p+1];
			for(int k=0; k<p;k++){
				auto_hidden_layer *a = preTrainLayers[k];
				activations[k] = new float[a->getNumHiddenUnits()];
				deltas[k] = new float[a->getNumHiddenUnits()];
				if(k==0){
					a->encode(o_i, activations[k]);
				}
				else{
					a->encode(activations[k-1], activations[k]);
				}
			}
			activations[p] = new float[output->getNumHiddenUnits()];
			output->encode(activations[p-1], activations[p]);
			deltas[p] = new float[output->getNumHiddenUnits()];
			// all activations computed
			
			// Now compute the deltas for the hidden/output layers (not the autoencoder layers)
			output->compute_delta_output(deltas[p], activations[p], trainLabels[j]);

  		output->updateWeights(deltas[p], activations[p-1], learn_rate);

			// get layers in order starting from the last one and we'll update in that order
			for (int k = numPreTrainLayers-1; k > 0; k--) // TODO should this really only go until 1?
			{
				auto_hidden_layer *a = preTrainLayers[k];
				// we use the compute delta hidden from the neural net as opposed to the
				// autoencoder since we don't want to compare the original with encoded/decoded images.
				if(k == p-1){
					a->compute_delta_hidden(deltas[k], deltas[k+1], activations[k], h);
				}
				else{
					a->compute_delta_hidden(deltas[k], deltas[k+1], activations[k], preTrainLayers[k+1]);
				}

				if(k == 0){ // if we're on the first hidden layer, the input is the images, not the previous layer
					a->updateWeights(deltas[k], o_i, preTrainLayersLearnRates[k]);
				}
				else{
					a->updateWeights(deltas[k], activations[k-1], preTrainLayersLearnRates[k]);
				}
				// Need to add fine tuning for the weights from the input to the first activation layer here.
			}
			
			// after updating all the weights we compute the error again
			for(int k=0; k<p;k++){
				auto_hidden_layer *a = preTrainLayers[k];
				if(k==0){
					a->encode(o_i, activations[k]);
				}
				else{
					a->encode(activations[k-1], activations[k]);
				}
			}
			output->encode(activations[p-1], activations[p]);

			sum_squared_error += o->squared_loss(activations[p+1], trainLabels[j]);

			// Free the memory that we allocated
			for(int k=0; k<p;k++){
				delete[] activations[k];
				delete[] deltas[k];
			}
			delete activations;
			delete deltas;
		}
		cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
	}


	return;
}



void autoencoder::testFineNoHidden(float **testingImages, vector<int> &testLabels, int numTestingImages)
{
  cout << "evaluating network on " << numTestingImages << " test digits" << endl;
  float correct = 0;
  for (int i = 0; i < numTestingImages; i++)
  {
    float *o_i = testingImages[i];
    getInput(o_i);
    int predict_value = predictFineNoHidden(o_i);
    if (predict_value == testLabels[i])
    {
      correct += 1;
    }
  }
  cout << "accuracy rate: " << correct / numTestingImages << endl;
}

int autoencoder::predictFineNoHidden(float *o_i)
{
	float out[output->getNumHiddenUnits()];
  output->encode(o_i, out);

  int bestIndex = -1;
  float bestValue = -1;
  for (int i = 0; i < (int) output->getNumHiddenUnits(); i++)
  {
    // cout << o_k[i] << " ";
    if (out[i] > bestValue)
    {
      bestValue = out[i];
      bestIndex = i;
    }
  }
  // cout << endl;
  return bestIndex;
}


