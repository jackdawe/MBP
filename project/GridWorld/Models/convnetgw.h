#ifndef CONVNETGW_H
#define CONVNETGW_H
#undef slots
#include "../toolsgw.h"
#include <torch/torch.h>
#define slots Q_SLOTS
#include <gflags/gflags.h>
DECLARE_int32(conv1);
DECLARE_int32(conv2);
DECLARE_int32(fc1);

class ConvNetGWImpl: public torch::nn::Module
{
public:
    ConvNetGWImpl();
    ConvNetGWImpl(int size, int nConv1, int nConv2, int nfc);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor actorOutput(torch::Tensor batch);
    torch::Tensor criticOutput(torch::Tensor batch);
    torch::Device getUsedDevice();
    
private:
    torch::Device usedDevice; 
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
