#include "convnetgw.h"

ConvNetGW::ConvNetGW():
  usedDevice(torch::Device(torch::kCPU)){}

ConvNetGW::ConvNetGW(int size, int nConv1, int nConv2, int nfc):
  usedDevice(torch::Device(torch::kCPU)),size(size),nConv1(nConv1), nConv2(nConv2), nfc(nfc)
{
    conv1 = register_module("conv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nConv1,3).stride(1).padding(1)));
    conv2 = register_module("conv2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1,nConv2,3).stride(1).padding(1)));
    fc = register_module("fc",torch::nn::Linear(nConv2*size*size/16,nfc));
    actor = register_module("actor",torch::nn::Linear(nfc,4));
    critic = register_module("critic",torch::nn::Linear(nfc,1));
    if (torch::cuda::is_available())
      {
	std::cout << "Training will be done using CUDA" << std::endl;
	usedDevice = torch::Device(torch::kCUDA);
      }
    else
      {
	std::cout <<"Training will be done using CPU"<<std::endl;
      }
    this->to(usedDevice);
}

torch::Tensor ConvNetGW::forward(torch::Tensor x)
{
    x = torch::relu(torch::max_pool2d(conv1->forward(x),2));
    x = torch::relu(torch::max_pool2d(conv2->forward(x),2));
    x = x.view({-1,nConv2*size*size/16});
    x = torch::relu(fc->forward(x));
    return x;
}

torch::Tensor ConvNetGW::actorOutput(torch::Tensor x)
{
    x = this->forward(x);
    return torch::softmax(actor->forward(x),1);
}

torch::Tensor ConvNetGW::criticOutput(torch::Tensor x)
{
    x = this->forward(x);
    return critic->forward(x);
}

torch::Device ConvNetGW::getUsedDevice()
{
  return usedDevice;
}
