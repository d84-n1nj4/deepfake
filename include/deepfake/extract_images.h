#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <torch/script.h>

// Function to return image read at location given as type torch::Tensor
// Resizes image to (224, 224, 3)
torch::Tensor read_data(std::string location);

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
torch::Tensor read_label(int label);

// Function returns vector of tensors (images) read from the list of images in a folder
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images);

// Function returns vector of tensors (labels) read from the list of labels
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels);

// Function to load data from given folder(s) name(s) (folders_name)
// Returns pair of vectors of string (image locations) and int (respective labels)
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name);

// Function to train the network on train data
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);

// Function to test the network on test data
template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size);

