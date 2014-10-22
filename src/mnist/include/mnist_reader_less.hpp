//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>
#include <cmath>

namespace mnist {

template<typename Pixel = uint8_t, typename Label = uint8_t>
struct MNIST_dataset {
    std::vector<std::vector<Pixel>> training_images;
    std::vector<std::vector<Pixel>> test_images;
    std::vector<Label> training_labels;
    std::vector<Label> test_labels;
};

inline uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

template<typename Pixel = uint8_t, typename Label = uint8_t>
std::vector<std::vector<Pixel>> read_mnist_image_file(const std::string& path){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = read_header(buffer, 0);

        if(magic != 0x803){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = read_header(buffer, 1);
            auto rows = read_header(buffer, 2);
            auto columns = read_header(buffer, 3);

            if(size < count * rows * columns + 16){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                //Cast to unsigned char is necessary cause signedness of char is
                //platform-specific
                auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

                std::vector<std::vector<Pixel>> images;
                images.reserve(count);

                for(size_t i = 0; i < count; ++i){
                    images.emplace_back(rows * columns);

                    for(size_t j = 0; j < rows * columns; ++j){
                        auto pixel = *image_buffer++;
                        images[i][j] = static_cast<Pixel>(pixel);
                    }
                }

                return std::move(images);
            }
        }
    }

    return {};
}


template<typename Pixel = uint8_t, typename Label = uint8_t>
std::vector<std::vector<std::vector<Pixel>>> read_mnist_image_file_sq(const std::string& path){
		std::vector<std::vector<std:vector<Pixel>>> sq_images;
		auto images =  file read_mnist_image_file(path);
		int size = std::sqrt(images[0].size());
		int num_images = images.size();
		sq_images.resize(num_images);
		for(int im=0;im<num_images;im++){
			sq_images[im].resize(size);
			for(int i=0;i<size;i++){
				sq_images[im][i].resize(size);
				for(int j=0;j<size;j++){
					sq_images[im][i][j] = images[im][i*size+j];
				}
			}
		}

    return sq_images;
}

template<typename Label = uint8_t>
std::vector<Label> read_mnist_label_file(const std::string& path){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = read_header(buffer, 0);

        if(magic != 0x801){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = read_header(buffer, 1);

            if(size < count + 8){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                //Cast to unsigned char is necessary cause signedness of char is
                //platform-specific
                auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

                std::vector<Label> labels(count);

                for(size_t i = 0; i < count; ++i){
                    auto label = *label_buffer++;
                    labels[i] = static_cast<Label>(label);
                }

                return std::move(labels);
            }
        }
    }

    return {};
}

} //end of namespace mnist

#endif
