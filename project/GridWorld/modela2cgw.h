#ifndef MODELA2CGW_H
#define MODELA2CGW_H
#include <torch/torch.h>

class ModelA2CGW: public torch::nn::Module
{
public:
    ModelA2CGW();
    ModelA2CGW(int nInputs,int nHidden1,int nHidden2,int nHidden3);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor actorOutput(torch::Tensor x);
    torch::Tensor criticOutput(torch::Tensor x);
private:
    int nInputs;
    int nHidden1;
    int nHidden2;
    int nHidden3;
    std::shared_ptr<torch::nn::LinearImpl> fc1;
    std::shared_ptr<torch::nn::LinearImpl> fc2;
    std::shared_ptr<torch::nn::LinearImpl> fc3;
    std::shared_ptr<torch::nn::LinearImpl> actor;
    std::shared_ptr<torch::nn::LinearImpl> critic;
};

#endif // MODELA2CGW_H
