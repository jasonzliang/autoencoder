#include "mnist/include/mnist_reader.hpp"
#include "neural_network.h"
#include "autoencoder.h"
#include "neural_network_cross.h"
#include <iomanip>
// #include <iostream>
// #include <vector>

using namespace std;

int numTrainingImages;
int numTestingImages;

void parseImages(vector<vector<vector<float> > > &__images, float **images, int numImages)
{
  for (int i = 0; i < numImages; i++ )
  {
    int counter = 0;
    for (int j = 0; j < 28; j++)
    {
      for (int k = 0; k < 28; k++)
      {
        images[i][counter] = __images[i][j][k] / 255.0;
        counter++;
      }
    }
  }
}

void train_and_test_network_cross(int numOuterIter, vector<int> &trainLabels, float **trainingImages, vector<int> &testLabels, float **testingImages)
{
  neural_network_cross *myNeuralNet = new neural_network_cross(784, 500, 10, 0.1);
  cout << "cycling through " << numTrainingImages << " training images for " << numOuterIter << " outer iterations" << endl;
  double start = omp_get_wtime();
  for (int i = 0; i < numOuterIter; i++ )
  {
    float sum_squared_error = 0.0;
    for (int j = 0; j < numTrainingImages; j++)
    {
      sum_squared_error += myNeuralNet->backprop(trainingImages[j], trainLabels[j]);
    }
    cout << "outer iter: " << i + 1 << " wall time: " << omp_get_wtime() - start << " total error: " << sum_squared_error << endl;
  }

  cout << "evaluating network on " << numTestingImages << " test digits" << endl;
  float correct = 0;
  for (int i = 0; i < numTestingImages; i++)
  {
    int predict_value = myNeuralNet->predict(testingImages[i]);
    if (predict_value == testLabels[i])
    {
      correct += 1;
    }
  }
  cout << "accuracy rate: " << correct / numTestingImages << endl;

  delete myNeuralNet;
}

void train_and_test_network_square(int numOuterIter, vector<int> &trainLabels, float **trainingImages, vector<int> &testLabels, float **testingImages)
{
  neural_network *myNeuralNet = new neural_network(784, 500, 10, 0.1);
  myNeuralNet->train(trainingImages, trainLabels, numOuterIter, numTrainingImages);
  myNeuralNet->test(testingImages, testLabels, numTestingImages);
  delete myNeuralNet;
}

void train_and_test_autoencoder(vector<int> &trainLabels, float **trainingImages, vector<int> &testLabels, float **testingImages)
{
  vector<int> autoencoder_layers {784, 1000, 1000};
  vector<float> auto_learn_rates {0.001, 0.001, 0.001};
  vector<int> auto_iters {15, 15, 15};
  vector<float> noise_levels {0.1, 0.2, 0.3};

  autoencoder *myAutoencoder = new autoencoder(autoencoder_layers, auto_learn_rates, auto_iters, noise_levels, 1000, 500, 10, 0.1);
  myAutoencoder->preTrain(trainingImages, numTrainingImages);
  myAutoencoder->train(trainingImages, trainLabels, 30, numTrainingImages);
  myAutoencoder->test(testingImages, testLabels, numTestingImages);
  delete myAutoencoder;
}

void experiment_1(vector<int> &trainLabels, float **trainingImages)
{
  cout << "running experiment 1" << endl;
  vector<int> autoencoder_layers {784};
  vector<float> auto_learn_rates {0.001};
  vector<int> auto_iters {15};
  vector<float> noise_levels {0.25};

  int cores[3] = {1, 4, 8};
  for (int i = 0; i < 3; i++)
  {
    omp_set_num_threads(cores[i]);
    autoencoder *myAutoencoder = new autoencoder(autoencoder_layers, auto_learn_rates, auto_iters, noise_levels, 500, 300, 10, 0.05);
    cout << "using " << cores[i] << " cores" << endl;;
    myAutoencoder->preTrain(trainingImages, 5000);
    delete myAutoencoder;
  }
}

void experiment_2(vector<int> &trainLabels, float **trainingImages)
{
  cout << "running experiment 2" << endl;
  vector<int> autoencoder_layers {784};
  vector<float> auto_learn_rates {0.001};
  vector<int> auto_iters {1};
  vector<float> noise_levels {0.25};

  int cores[3] = {1, 4, 8};
  int num_hidden_units[4] = {100, 200, 500, 800};
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      omp_set_num_threads(cores[j]);
      autoencoder *myAutoencoder = new autoencoder(autoencoder_layers, auto_learn_rates, auto_iters, noise_levels, num_hidden_units[i], 300, 10, 0.05);
      cout << "using " << num_hidden_units[i] << " hidden units and " << cores[j] << " cores" << endl;
      myAutoencoder->preTrain(trainingImages, 60000);
      delete myAutoencoder;
    }
  }
}

void experiment_3(vector<int> &trainLabels, float **trainingImages, vector<int> &testLabels, float **testingImages)
{
  omp_set_num_threads(omp_get_num_procs());
  cout << "running experiment 3" << endl;
  vector<int> autoencoder_layers {784};
  vector<float> auto_learn_rates {0.001};
  vector<int> auto_iters {15};
  vector<float> noise_levels {0.25};
  autoencoder *myAutoencoder = new autoencoder(autoencoder_layers, auto_learn_rates, auto_iters, noise_levels, 500, 300, 10, 0.05);

  myAutoencoder->preTrain(trainingImages, 60000);
  myAutoencoder->visualizeWeights(0, 100);
  myAutoencoder->reconstructImage(testingImages, 0, 100);

  delete myAutoencoder;
}

int main(int argc, char *argv[])
{
  cout << "using " << omp_get_num_procs() << " cores" << endl;
  omp_set_num_threads(omp_get_num_procs());
  string train_labels_file("../data/train-labels-idx1-ubyte");
  string test_labels_file("../data/t10k-labels-idx1-ubyte");
  string train_images_file("../data/train-images-idx3-ubyte");
  string test_images_file("../data/t10k-images-idx3-ubyte");


  // Read in the data, by default the functions read in the values
  // as unsigned 8bit ints (uint8_t). read_mnist_label_file reads
  // data into a vector by default (contained may be changed).
  // read_mnist_image_file reads in the images as a vector (1D) and
  // read_mnist_image_file_sq reads the images in as a 2D vector
  vector<int> training_labels = mnist::read_mnist_label_file<vector, int>(train_labels_file);
  vector<vector<vector<float> > > training_images_sq = mnist::read_mnist_image_file_sq<vector, vector, float>(train_images_file);
  numTrainingImages = (int) training_images_sq.size();

  float **training_images = new float*[numTrainingImages];
  for (int i = 0; i < numTrainingImages; i++)
  {
    training_images[i] = new float[784];
  }
  parseImages(training_images_sq, training_images, numTrainingImages);

  vector<int> testing_labels = mnist::read_mnist_label_file<vector, int>(test_labels_file);
  vector<vector<vector<float> > > testing_images_sq = mnist::read_mnist_image_file_sq<vector, vector, float>(test_images_file);
  numTestingImages = (int) testing_images_sq.size();
  float **testing_images = new float*[numTestingImages];
  for (int i = 0; i < numTestingImages; i++)
  {
    testing_images[i] = new float[784];
  }
  parseImages(testing_images_sq, testing_images, numTestingImages);

  cout << "finished parsing input data! " << endl;

  // train_and_test_network_square(30, training_labels, training_images, testing_labels, testing_images);
  // train_and_test_network_cross(30, training_labels, training_images, testing_labels, testing_images);
  train_and_test_autoencoder(training_labels, training_images, testing_labels, testing_images);
  // experiment_1(training_labels, training_images);
  // experiment_2(training_labels, training_images);
  // experiment_3(training_labels, training_images, testing_labels, testing_images);

  for (int i = 0; i < numTrainingImages; i++)
  {
    delete[] training_images[i];
  }
  delete[] training_images;

  for (int i = 0; i < numTestingImages; i++)
  {
    delete[] testing_images[i];
  }
  delete[] testing_images;

  return 0;
}

// for (int i = 0; i < 10; i++)
// {
//   cout << "label: " << training_labels[i] << endl;
//   for (int j = 0; j < 28; j++)
//   {
//     for (int k = 0; k < 28; k++)
//     {
//       if (training_images[i][j * 28 + k] > 0.5)
//       {
//         cout << "#";
//       }
//       else
//       {
//         cout << " ";
//       }
//     }
//     cout << endl;
//   }
// }
