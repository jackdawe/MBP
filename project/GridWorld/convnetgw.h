#ifndef CONVNETGW_H
#define CONVNETGW_H
#include <torch/torch.h>

class ConvNetGW: torch::nn::Module
{
public:
    ConvNetGW();
    ConvNetGW(int size, int nConv1, int nConv2, int nfc);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor actorOutput(torch::Tensor x);
    torch::Tensor criticOutput(torch::Tensor x);

private:
    int size;
    int nConv1;
    int nConv2;
    int nfc;
    std::shared_ptr<torch::nn::Conv2dImpl> conv1;
    std::shared_ptr<torch::nn::Conv2dImpl> conv2;
    std::shared_ptr<torch::nn::LinearImpl> fc;
    std::shared_ptr<torch::nn::LinearImpl> actor;
    std::shared_ptr<torch::nn::LinearImpl> critic;
};

#endif // CONVNETGW_H
