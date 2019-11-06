#include "modela2cgw.h"

ModelA2CGW::ModelA2CGW():
  usedDevice(torch::Device(torch::kCPU))
{    
}

ModelA2CGW::ModelA2CGW(int nInputs, int nHidden1, int nHidden2, int nHidden3):
  usedDevice(torch::Device(torch::kCPU)),nInputs(nInputs),nHidden1(nHidden1),nHidden2(nHidden2),nHidden3(nHidden3)
{
    fc1 = register_module("fc1",torch::nn::Linear(nInputs,nHidden1));
    fc2 = register_module("fc2",torch::nn::Linear(nHidden1,nHidden2));
    fc3 = register_module("fc3",torch::nn::Linear(nHidden2,nHidden3));
    actor = register_module("actor",torch::nn::Linear(nHidden3,4));
    critic = register_module("critic",torch::nn::Linear(nHidden3,1));
}

torch::Tensor ModelA2CGW::forward(torch::Tensor x)
{
    x = torch::relu(fc1->forward(x.reshape({x.size(0),nInputs})));
    x = torch::relu(fc2->forward(x));
    return torch::relu(fc3->forward(x));
}

torch::Tensor ModelA2CGW::actorOutput(torch::Tensor x)
{
    x = forward(x);
    return torch::softmax(actor->forward(x),1);
}

torch::Tensor ModelA2CGW::criticOutput(torch::Tensor x)
{
    x = forward(x);
    return critic->forward(x);
}

torch::Device ModelA2CGW::getUsedDevice()
{
  return usedDevice;
}

