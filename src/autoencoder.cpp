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
  output_l = new hidden_layer(preTrainLayerWidths[numPreTrainLayers - 1], 10);

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

void autoencoder::setO_i(float **o_i, float **trainingImages, int k, int numTrainingImages, int length)
{
  for (int i = 0; i < numTrainingImages; i++)
  {
    o_i[i] = new float[length];
  }

  for (int i = 0; i < numTrainingImages; i++)
  {
    float *o_t = trainingImages[i];
    getInputK(o_t, k);
    for (int j = 0; j < length; j++)
    {
      o_i[i][j] = o_t[j];
    }
  }
}

void autoencoder::setCorruptedO_i(float **corrupted_o_i, float **o_i, int k, int numTrainingImages, int length)
{
  for (int i = 0; i < numTrainingImages; i++)
  {
    corrupted_o_i[i] = new float[length];
    corrupt_masking(o_i[i], corrupted_o_i[i], preTrainLayersNoiseLevels[k], length);
  }
}

void autoencoder::deleteO_i(float **o_i, int numTrainingImages)
{
  for (int i = 0; i < numTrainingImages; i++)
  {
    delete[] o_i[i];
  }
  delete[] o_i;
}

void autoencoder::preTrain(float **trainingImages, int numTrainingImages)
{
  for (int k = 0; k < numPreTrainLayers; k++)
  {

    cout << "pretraining layer #" << k + 1 << ", cycling through " << numTrainingImages << " training images for " << preTrainLayersOuterIter[k] << " outer iterations" << endl;

    auto_hidden_layer *a = preTrainLayers[k];

    float **o_i = new float*[numTrainingImages];
    float **corrupted_o_i = new float*[numTrainingImages];
    if (k > 0)
    {
      setO_i(o_i, trainingImages, k, numTrainingImages, a->getNumInputUnits());
    }
    else
    {
      o_i = trainingImages;
    }
    setCorruptedO_i(corrupted_o_i, o_i, k, numTrainingImages, a->getNumHiddenUnits());

    float *o_e = new float[a->getNumHiddenUnits()];
    float *delta_e = new float[a->getNumHiddenUnits()];
    float *o_d = new float[a->getNumInputUnits()];
    float *delta_d = new float[a->getNumInputUnits()];

    float sum_squared_error = 0.0;
    for (int i = 0; i < numTrainingImages; i++)
    {
      float *o_t = o_i[i];
      getInputK(o_t, k);

      a->encode(o_t, o_e);
      a->decode(o_e, o_d);

      sum_squared_error += a->auto_squared_loss(o_t, o_d);
    }
    cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

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
        float *o_t = o_i[j];
        float *corrupted_o_t = corrupted_o_i[j];

        // t = omp_get_wtime();
        a->encode(corrupted_o_t, o_e);
        // total_encode_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->decode(o_e, o_d);
        // total_decode_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->auto_compute_delta_output(delta_d, o_d, o_t);
        // total_compute_delta_output_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->auto_compute_delta_hidden(delta_e, delta_d, o_e);
        // total_compute_delta_hidden_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        a->auto_updateWeights(delta_e, corrupted_o_t, delta_d, o_e, preTrainLayersLearnRates[k]);
        // total_updateWeights_time += omp_get_wtime() - t;

        // t = omp_get_wtime();
        sum_squared_error += a->auto_squared_loss(o_t, o_d);
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
    }

    if (k > 0)
    {
      deleteO_i(o_i, numTrainingImages);
    }
    deleteO_i(corrupted_o_i, numTrainingImages);

    delete[] o_e;
    delete[] o_d;
    delete[] delta_e;
    delete[] delta_d;
  }
}

void autoencoder::preTrainGAMiniBatch(float **trainingImages, int numTrainingImages)
{
  int batchSize = 5;

  for (int k = 0; k < numPreTrainLayers; k++)
  {

    cout << "pretraining layer #" << k + 1 << " with GA, cycling through " << numTrainingImages << " training images for " << preTrainLayersOuterIter[k] << " outer iterations" << endl;

    auto_hidden_layer *a = preTrainLayers[k];

    float **o_i = new float*[numTrainingImages];
    float **corrupted_o_i = new float*[numTrainingImages];
    if (k > 0)
    {
      setO_i(o_i, trainingImages, k, numTrainingImages, a->getNumInputUnits());
    }
    else
    {
      o_i = trainingImages;
    }
    setCorruptedO_i(corrupted_o_i, o_i, k, numTrainingImages, a->getNumHiddenUnits());

    float *o_e = new float[a->getNumHiddenUnits()];
    float *o_d = new float[a->getNumInputUnits()];

    float sum_squared_error = 0.0;
    for (int i = 0; i < numTrainingImages; i++)
    {
      float *o_t = o_i[i];
      getInputK(o_t, k);
      a->encode(o_t, o_e);
      a->decode(o_e, o_d);
      sum_squared_error += a->auto_squared_loss(o_t, o_d);
    }
    cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

    double start = omp_get_wtime();

    //setup GA and its parameters
    ga_params myParams;
    myParams.mutRate = 0.0001;
    myParams.mutAmount = 0.1 * a->getWeightRange();
    myParams.crossRate = 0.5;
    myParams.popSize = 10;
    myParams.genomeSize = a->getNumWeights() + a->getNumInputUnits() + a->getNumHiddenUnits();
    myParams.numWeights = a->getNumWeights();
    myParams.numToReplace = 4;
    myParams.initRange = 1.0 * a->getWeightRange();
    myParams.alpha = 0.5;
    myParams.chunkSize = 15000;

    genetic *ga = new genetic(myParams);

    delete[] a->getWeights();
    delete[] a->getEncodeBiases();
    delete[] a->getDecodeBiases();
    int encodeBiasesOffset = a->getNumWeights();
    int decodeBiasesOffset = encodeBiasesOffset + a->getNumHiddenUnits();

    for (int i = 0; i < preTrainLayersOuterIter[k]; i++ )
    {

      float sum_squared_error = 0.0;
      for (int j = 0; j < numTrainingImages; j += batchSize)
      {
        for (int currInd = 0; currInd < myParams.popSize; currInd++)
        {
          float *genome = ga->getGenome(currInd);
          a->setWeights(genome);
          a->setEncodeBiases(&(genome[encodeBiasesOffset]));
          a->setDecodeBiases(&(genome[decodeBiasesOffset]));

          float error = 0.0;
          for (int currBatch = j; currBatch < j + batchSize; currBatch++)
          {
            int __currBatch = currBatch % numTrainingImages;
            float *o_t = o_i[__currBatch];
            float *corrupted_o_t = corrupted_o_i[__currBatch];
            a->encode(corrupted_o_t, o_e);
            a->decode(o_e, o_d);
            error += a->auto_squared_loss(o_t, o_d);
          }
          ga->setFitness(currInd, 1.0 / log((error * numTrainingImages) / batchSize));
          // ga->setFitness(currInd, 1.0 / error);

          if (currInd == myParams.popSize - 1)
          {
            sum_squared_error += error;
          }
        }

        ga->step();
        // ga->printStats();
      }
      cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
    }

    //cleanup
    if (k > 0)
    {
      deleteO_i(o_i, numTrainingImages);
    }
    deleteO_i(corrupted_o_i, numTrainingImages);

    delete[] o_e;
    delete[] o_d;

    delete ga;
  }
}

/*  good parameters

    myParams.mutRate = 0.00005;
    myParams.mutAmount = 0.1 * a->getWeightRange();
    myParams.crossRate = 0.5;
    myParams.popSize = 10;
    myParams.genomeSize = a->getNumWeights() + a->getNumInputUnits() + a->getNumHiddenUnits();
    myParams.numWeights = a->getNumWeights();
    myParams.numToReplace = 6;
    myParams.initRange = 1.0 * a->getWeightRange();
    myParams.alpha = 1.0;
    myParams.chunkSize = 15000;
*/

void autoencoder::preTrainGA(float **trainingImages, int numTrainingImages)
{
  for (int k = 0; k < numPreTrainLayers; k++)
  {

    cout << "pretraining layer #" << k + 1 << " with GA, cycling through " << numTrainingImages << " training images for " << preTrainLayersOuterIter[k] << " outer iterations" << endl;

    auto_hidden_layer *a = preTrainLayers[k];

    float **o_i = new float*[numTrainingImages];
    float **corrupted_o_i = new float*[numTrainingImages];
    if (k > 0)
    {
      setO_i(o_i, trainingImages, k, numTrainingImages, a->getNumInputUnits());
    }
    else
    {
      o_i = trainingImages;
    }
    setCorruptedO_i(corrupted_o_i, o_i, k, numTrainingImages, a->getNumHiddenUnits());

    float *o_e = new float[a->getNumHiddenUnits()];
    float *o_d = new float[a->getNumInputUnits()];

    float sum_squared_error = 0.0;
    for (int i = 0; i < numTrainingImages; i++)
    {
      float *o_t = o_i[i];
      getInputK(o_t, k);
      a->encode(o_t, o_e);
      a->decode(o_e, o_d);
      sum_squared_error += a->auto_squared_loss(o_t, o_d);
    }
    cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

    double start = omp_get_wtime();

    //setup GA and its parameters
    ga_params myParams;
    myParams.mutRate = 0.00005;
    myParams.mutAmount = 0.1 * a->getWeightRange();
    myParams.crossRate = 0.5;
    myParams.popSize = 10;
    myParams.genomeSize = a->getNumWeights() + a->getNumInputUnits() + a->getNumHiddenUnits();
    myParams.numWeights = a->getNumWeights();
    myParams.numToReplace = 6;
    myParams.initRange = 1.0 * a->getWeightRange();
    myParams.alpha = 1.0;
    myParams.chunkSize = 15000;

    genetic *ga = new genetic(myParams);

    delete[] a->getWeights();
    delete[] a->getEncodeBiases();
    delete[] a->getDecodeBiases();
    int encodeBiasesOffset = a->getNumWeights();
    int decodeBiasesOffset = encodeBiasesOffset + a->getNumHiddenUnits();

    for (int i = 0; i < preTrainLayersOuterIter[k]; i++ )
    {

      float sum_squared_error = 0.0;
      for (int j = 0; j < numTrainingImages; j++)
      {
        float *o_t = o_i[j];
        float *corrupted_o_t = corrupted_o_i[j];

        for (int currInd = 0; currInd < myParams.popSize; currInd++)
        {
          // if (currInd >= myParams.numToReplace and currInd != myParams.popSize - 1)
          // {
          //   continue;
          // }

          float *genome = ga->getGenome(currInd);
          a->setWeights(genome);
          a->setEncodeBiases(&(genome[encodeBiasesOffset]));
          a->setDecodeBiases(&(genome[decodeBiasesOffset]));
          a->encode(corrupted_o_t, o_e);
          a->decode(o_e, o_d);
          float error = a->auto_squared_loss(o_t, o_d);
          ga->setFitness(currInd, 1.0 / log(error * numTrainingImages));

          if (currInd == myParams.popSize - 1)
          {
            sum_squared_error += error;
          }
        }

        ga->step();
        // ga->printStats();
      }
      cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
    }

    //cleanup
    if (k > 0)
    {
      deleteO_i(o_i, numTrainingImages);
    }
    deleteO_i(corrupted_o_i, numTrainingImages);

    delete[] o_e;
    delete[] o_d;

    delete ga;
  }
}

void autoencoder::train(float **trainingImages, vector<int> &trainLabels, int numOuterIter, int numTrainingImages)
{
  cout << "cycling through " << numTrainingImages << " training images for " << numOuterIter << " outer iterations" << endl;

  float **o_i = new float*[numTrainingImages];
  setO_i(o_i, trainingImages, numPreTrainLayers, numTrainingImages, h->getNumInputUnits());

  float sum_squared_error = 0.0;
  for (int j = 0; j < numTrainingImages; j++)
  {
    float *_o_i = o_i[j];

    h->encode(_o_i, o_j);
    o->encode(o_j, o_k);

    sum_squared_error += o->squared_loss(o_k, trainLabels[j]);
  }
  cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;

  double start = omp_get_wtime();
  for (int i = 0; i < numOuterIter; i++ )
  {
    float sum_squared_error = 0.0;
    for (int j = 0; j < numTrainingImages; j++)
    {
      float *_o_i = o_i[j];

      sum_squared_error += backprop(_o_i, trainLabels[j]);
    }
    cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
  }

  deleteO_i(o_i, numTrainingImages);
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

void autoencoder::fineTune(float **trainingImages, int numTrainingImages, vector<int> &trainLabels)
{
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
      activations =  new float*[p + 2];
      deltas =  new float*[p + 2];
      for (int k = 0; k < p; k++)
      {
        auto_hidden_layer *a = preTrainLayers[k];
        activations[k] = new float[a->getNumHiddenUnits()];
        deltas[k] = new float[a->getNumHiddenUnits()];
        if (k == 0)
        {
          a->encode(o_i, activations[k]);
        }
        else
        {
          a->encode(activations[k - 1], activations[k]);
        }
      }
      activations[p] = new float[h->getNumHiddenUnits()];
      h->encode(activations[p - 1], activations[p]);
      deltas[p] = new float[h->getNumHiddenUnits()];
      activations[p + 1] = new float[o->getNumHiddenUnits()];
      o->encode(activations[p], activations[p + 1]);
      deltas[p + 1] = new float[o->getNumHiddenUnits()];
      // all activations computed

      // Now compute the deltas for the hidden/output layers (not the autoencoder layers)
      o->compute_delta_output(deltas[p + 1], activations[p + 1], trainLabels[j]);
      h->compute_delta_hidden(deltas[p], deltas[p + 1], activations[p], o);

      o->updateWeights(deltas[p + 1], activations[p], learn_rate);
      h->updateWeights(deltas[p], activations[p - 1], learn_rate);

      // get layers in order starting from the last one and we'll update in that order
      for (int k = numPreTrainLayers - 1; k > 1; k--) // TODO should this really only go until 1?
      {
        auto_hidden_layer *a = preTrainLayers[k];
        // we use the compute delta hidden from the neural net as opposed to the
        // autoencoder since we don't want to compare the original with encoded/decoded images.
        if (k == p - 1)
        {
          a->compute_delta_hidden(deltas[k], deltas[k + 1], activations[k], h);
        }
        else
        {
          a->compute_delta_hidden(deltas[k], deltas[k + 1], activations[k], preTrainLayers[k + 1]);
        }
        a->updateWeights(deltas[k], activations[k - 1], preTrainLayersLearnRates[k]);
        // Need to add fine tuning for the weights from the input to the first activation layer here.
      }

      // after updating all the weights we compute the error again
      for (int k = 0; k < p; k++)
      {
        auto_hidden_layer *a = preTrainLayers[k];
        if (k == 0)
        {
          a->encode(o_i, activations[k]);
        }
        else
        {
          a->encode(activations[k - 1], activations[k]);
        }
      }
      h->encode(activations[p - 1], activations[p]);
      o->encode(activations[p], activations[p + 1]);

      sum_squared_error += o->squared_loss(activations[p + 1], trainLabels[j]);

      // Free the memory that we allocated
      for (int k = 0; k < p; k++)
      {
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


void autoencoder::fineTuneNoHidden(float **trainingImages, int numTrainingImages, vector<int> &trainLabels)
{
  cout << "STARTING THE FINE TUNING STEP" << endl;

  int numOuterIter = 3;

  cout << "cycling through " << numTrainingImages << " training images for " << numOuterIter << " outer iterations" << endl;
  float sum_squared_error = 0.0;

  // Forward activiate the network to compute the initial output error
  float *initac = new float[output_l->getNumHiddenUnits()];
  for (int j = 0; j < numTrainingImages; j++)
  {
    float *o_i = trainingImages[j];
    getInput(o_i);


    output_l->encode(o_i, initac);
    sum_squared_error += o->squared_loss(initac, trainLabels[j]);
  }
  delete[] initac;

  cout << "outer iter: 0 wall time: 0.00000 total error: " << sum_squared_error << endl;


  double start = omp_get_wtime();
  for (int i = 0; i < numOuterIter; i++ ) // TODO should make this an input probably
  {
    float sum_squared_error = 0.0;
    int p = numPreTrainLayers;
    //cout << "p: " << p << endl;
    for (int j = 0; j < numTrainingImages; j++)
      //for (int j = 0; j < 1; j++)
    {
      // compute all forward activations
      float *o_i = trainingImages[j];
      float **activations;
      float **deltas;
      activations =  new float*[p + 1];
      deltas =  new float*[p + 1];

      //auto_hidden_layer *a = preTrainLayers[0];
      //activations[0] = new float[a->getNumHiddenUnits()];
      //deltas[0] = new float[a->getNumHiddenUnits()];
      //a->encode(o_i, activations[0]);

      for (int k = 0; k < p; k++)
      {
        auto_hidden_layer *a = preTrainLayers[k];
        activations[k] = new float[a->getNumHiddenUnits()];
        deltas[k] = new float[a->getNumHiddenUnits()];
        if (k == 0)
        {
          a->encode(o_i, activations[k]);
        }
        else
        {
          a->encode(activations[k - 1], activations[k]);
        }
      }

      activations[p] = new float[output_l->getNumHiddenUnits()];
      output_l->encode(activations[p - 1], activations[p]);
      deltas[p] = new float[output_l->getNumHiddenUnits()];
      // all activations computed

      // Now compute the deltas for the hidden/output layers (not the autoencoder layers)
      output_l->compute_delta_output(deltas[p], activations[p], trainLabels[j]);
      output_l->updateWeights(deltas[p], activations[p - 1], learn_rate);

      if (p >= 2)
      {
        preTrainLayers[p - 1]->compute_delta_hidden(deltas[p - 1], deltas[p], activations[p - 1], output_l);
        preTrainLayers[p - 1]->updateWeights(deltas[p - 1], activations[p - 2], preTrainLayersLearnRates[p - 1]);

        // get layers in order starting from the last one and we'll update in that order
        for (int k = numPreTrainLayers - 2; k > 0; k--) // TODO should this really only go until 1?
        {
          auto_hidden_layer *a = preTrainLayers[k];
          // we use the compute delta hidden from the neural net as opposed to the
          // autoencoder since we don't want to compare the original with encoded/decoded images.
          a->compute_delta_hidden(deltas[k], deltas[k + 1], activations[k], preTrainLayers[k + 1]);
          a->updateWeights(deltas[k], activations[k - 1], preTrainLayersLearnRates[k]);
        }
      }

      preTrainLayers[0]->compute_delta_hidden(deltas[0], deltas[1], activations[0], preTrainLayers[1]);
      //preTrainLayers[0]->compute_delta_hidden(deltas[0], deltas[1], activations[0], output_l);
      preTrainLayers[0]->updateWeights(deltas[0], o_i, preTrainLayersLearnRates[0]);

      // after updating all the weights we compute the error again
      /*
      for(int k=0; k<p;k++){
        cout << "here?!?!?" << endl;
        auto_hidden_layer *a = preTrainLayers[k];
        if(k==0){
          a->encode(o_i, activations[k]);
        }
        else{
          a->encode(activations[k-1], activations[k]);
        }
      }
      */
      getInput(o_i);
      output_l->encode(o_i, activations[p]);
      //output_l->encode(activations[p-1], activations[p]);

      //for(int l=0;l< output_l->getNumHiddenUnits();l++){
      //  cout << activations[p][l] <<endl;
      //}
      sum_squared_error += output_l->squared_loss(activations[p], trainLabels[j]);
      //cout << "j: " << j << " error: " << sum_squared_error << " label: "<< trainLabels[j] << endl;

      // Free the memory that we allocated
      for (int k = 0; k <= p; k++)
      {
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
  float out[output_l->getNumHiddenUnits()];
  output_l->encode(o_i, out);

  int bestIndex = -1;
  float bestValue = -1;
  for (int i = 0; i < (int) output_l->getNumHiddenUnits(); i++)
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


