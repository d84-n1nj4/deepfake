#define main_h

#include "/home/mnuppnau/deepfake/include/deepfake/extract_images.h"
#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>

// Function to train the network on train data
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);

// Function to test the network on test data
template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size);

// Custom Dataset class
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
public:
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        states = process_images(list_images);
        labels = process_labels(list_labels);
        ds_size = states.size();
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    
    torch::optional<size_t> size() const override {
        return ds_size;
    };
};
