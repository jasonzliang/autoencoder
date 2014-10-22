#include "mnist/include/mnist_reader.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char* argv[]){
	std::string train_labels_file("../data/train-labels-idx1-ubyte");
	std::string test_labels_file("../data/t10k-labels-idx1-ubyte");
	std::string train_images_file("../data/train-images-idx3-ubyte");
	std::string test_images_file("../data/t10k-images-idx3-ubyte");


	// Read in the data, by default the functions read in the values 
	// as unsigned 8bit ints (uint8_t). read_mnist_label_file reads
	// data into a vector by default (contained may be changed).
	// read_mnist_image_file reads in the images as a vector (1D) and
	// read_mnist_image_file_sq reads the images in as a 2D vector
	auto training_labels = mnist::read_mnist_label_file(train_labels_file);
	auto test_labels = mnist::read_mnist_label_file(test_labels_file);

	auto training_images_sq = mnist::read_mnist_image_file_sq(train_images_file);
	auto test_images_sq = mnist::read_mnist_image_file_sq(test_images_file);


	// loop through the first 10 training examples and then print them
	// to stdout.
	for(int idx=0;idx<10;idx++){
		auto im = training_images_sq[idx];
		std::cout << int(training_labels[idx]) << std::endl;
		for(auto rows: im){
			for(auto elem: rows){
				if(elem > 128){
					std::cout << "\u2593" << "\u2593";
				}
				else
					std::cout << "  ";
			}
			std::cout << "\n";
		}
	}

	return 0;
}

