#include "convnetgw.h"
DEFINE_int32(conv1,16,"The number of units in the first convolutionnal layer");
DEFINE_int32(conv2,16,"The number of units in the second convolutionnal layer");
DEFINE_int32(fc1,128,"THe number of units in the first fully connected layer");

ConvNetGWImpl::ConvNetGWImpl():
  usedDevice(torch::Device(torch::kCPU))
{
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

ConvNetGWImpl::ConvNetGWImpl(int size, int nConv1, int nConv2, int nfc):
  usedDevice(torch::Device(torch::kCPU)),size(size),nConv1(nConv1), nConv2(nConv2), nfc(nfc)
{
    conv1 = register_module("conv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nConv1,3).stride(1).padding(1)));
    conv2 = register_module("conv2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1,nConv2,3).stride(1).padding(1)));
    fc = register_module("fc",torch::nn::Linear(nConv2*size*size/16,nfc));
    fca = register_module("fca",torch::nn::Linear(nfc,64));
    fcc = register_module("fcc",torch::nn::Linear(nfc,64));
    actor = register_module("actor",torch::nn::Linear(64,4));
    critic = register_module("critic",torch::nn::Linear(64,1));
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

torch::Tensor ConvNetGWImpl::forward(torch::Tensor x)
{
    x = torch::relu(torch::max_pool2d(conv1->forward(x),2));
    x = torch::relu(torch::max_pool2d(conv2->forward(x),2));
    x = x.view({-1,nConv2*size*size/16});
    x = torch::relu(fc->forward(x));
    return x;
}

torch::Tensor ConvNetGWImpl::actorOutput(torch::Tensor batch)
{
  torch::Tensor x = ToolsGW().toRGBTensor(batch).to(usedDevice); 
  x = this->forward(x);
  x = fca->forward(x);
  x = torch::relu(x);
  return torch::softmax(actor->forward(x),1);
}

torch::Tensor ConvNetGWImpl::criticOutput(torch::Tensor batch)
{
  torch::Tensor x = ToolsGW().toRGBTensor(batch).to(usedDevice); 
  x = this->forward(x);
  x = fca->forward(x);
  x = torch::relu(x);
  return torch::tanh(critic->forward(x));
}

torch::Device ConvNetGWImpl::getUsedDevice()
{
  return usedDevice;
}
