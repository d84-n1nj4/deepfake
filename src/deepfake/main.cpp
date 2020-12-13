#include </home/mnuppnau/deepfake/include/deepfake/model.h>
#include </home/mnuppnau/deepfake/src/deepfake/main.h>

template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size) {
    /*
     This function trains the network on our data loader using optimizer.
     
     Also saves the model as model.pt after every epoch.
     Parameters
     ===========
     1. net (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. lin (torch::nn::Linear type) - last FC layer with revised out_features depending on the number of classes
     3. data_loader (DataLoader& type) - Training data loader
     4. optimizer (torch::optim::Optimizer& type) - Optimizer like Adam, SGD etc.
     5. size_t (dataset_size type) - Size of training dataset
     
     Returns
     ===========
     Nothing (void)
     */
    float best_accuracy = 0.0; 
    int batch_index = 0;
    
    for(int i=0; i<25; i++) {
        float mse = 0;
        float Acc = 0.0;
        
        for(auto& batch: *data_loader) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            
            // Should be of length: batch_size
            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);
            
            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();
            
            auto output = net.forward(input).toTensor();
            // For transfer learning
            output = output.view({output.size(0), -1});
            output = lin(output);
            
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
            
            loss.backward();
            optimizer.step();
            
            auto acc = output.argmax(1).eq(target).sum();
            
            Acc += acc.template item<float>();
            mse += loss.template item<float>();
            
            batch_index += 1;
        }

        mse = mse/float(batch_index); // Take mean of loss
        std::cout << "Epoch: " << i  << ", " << "Accuracy: " << Acc/dataset_size << ", " << "MSE: " << mse << std::endl;

        test(net, lin, data_loader, dataset_size);

        if(Acc/dataset_size > best_accuracy) {
            best_accuracy = Acc/dataset_size;
            std::cout << "Saving model" << std::endl;
            net.save("model.pt");
            torch::save(lin, "model_linear.pt");
        }
    }
}

template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size) {
    /*
     Function to test the network on test data
     
     Parameters
     ===========
     1. network (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. lin (torch::nn::Linear type) - last FC layer with revised out_features depending on the number of classes
     3. loader (Dataloader& type) - test data loader
     4. data_size (size_t type) - test data size
     
     Returns
     ===========
     Nothing (void)
     */
    network.eval();
    
    float Loss = 0, Acc = 0;
    
    for (const auto& batch : *loader) {
        auto data = batch.data;
        auto targets = batch.target.squeeze();
        
        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);

        std::vector<torch::jit::IValue> input;
        input.push_back(data);

        auto output = network.forward(input).toTensor();
        output = output.view({output.size(0), -1});
        output = lin(output);
        
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    
    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

int main(int argc, const char * argv[]) {
    // Set folder names for cat and dog images
    std::string train_video_dir = "/mnt/data/train/dfdc_train_part_0/";
    
    std::vector<std::string> folders_name;
    folders_name.push_back(train_video_dir);
    
    // Get paths of images and labels as int from the folder paths
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(folders_name);
    
    std::vector<std::string> list_images = pair_images_labels.first;
    std::vector<int> list_labels = pair_images_labels.second;
    
    // Initialize CustomDataset class and read data
    auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());

    // Load pre-trained model
    // You can also use: auto module = torch::jit::load(argv[1]);
    torch::jit::script::Module module = torch::jit::load(argv[1]);
    
    // Create new net
    auto net = std::make_shared<Net>();
    // Resource: https://discuss.pytorch.org/t/how-to-load-the-prebuilt-resnet-models-or-any-other-prebuilt-models/40269/8
    // For VGG: 512 * 14 * 14, 2

    torch::nn::Linear lin(512, 2); // the last layer of resnet, which we want to replace, has dimensions 512x1000
    torch::optim::Adam opt(lin->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset), 4);

    train(module, lin, data_loader, opt, custom_dataset.size().value());
    return 0;
} 
