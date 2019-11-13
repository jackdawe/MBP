#include "worldmodelgw.h"
DEFINE_int32(sc1,8,"'State Conv 1': The number of feature maps in the first layer of the Unet encoder. Layer 2 will have twice as many feature maps and so on.");
DEFINE_int32(afc1,32,"'Action Fully Connected 1': The number of hidden units in the first layer of the MLP that maps action vectors to action vector embeddings");
DEFINE_int32(afc2,64,"'Action Fully Connected 2': The number of hidden units in the second layer of the MLP that maps action vectors to action vector embeddings");
DEFINE_int32(rc1,8,"'Reward Conv 1': the number of feature maps in the first layer of the ConvNet that maps predicted states to rewards");
DEFINE_int32(rc2,16,"'Reward Conv 2': the number of feature maps in the second layer of the ConvNet that maps predicted states to rewards");
DEFINE_int32(rfc,64,"'Reward Fully Connected': the number of hidden units in the fully connected layer of the ConvNet that maps predicted states to predicted rewards.");


WorldModelGWImpl::WorldModelGWImpl():
  usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for WorldModelGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout <<"Training will be done using CPU"<<std::endl;
    }
  this->to(usedDevice);
}

WorldModelGWImpl::WorldModelGWImpl(int size,int nStateConv1, int nActionfc1, int nActionfc2):
  usedDevice(torch::Device(torch::kCPU)), size(size), nStateConv1(nStateConv1), nActionfc1(nActionfc1), nActionfc2(nActionfc2)
{

  //Switching to CUDA if available
  
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for WorldModelGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout <<"Training will be done using CPU"<<std::endl;
    }
  this->to(usedDevice);

  nUnetLayers = -1+log(size)/log(2); 
  stateConvLayers.push_back(register_module("stateConv1_"+std::to_string(1),torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nStateConv1,3).stride(1).padding(1))));
  stateConvLayers.push_back(register_module("stateConv2_"+std::to_string(2),torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1,nStateConv1,3).stride(1).padding(1))));
  for (int i=1;i<nUnetLayers;i++)
    {
      stateConvLayers.push_back(register_module("stateConv1_"+std::to_string(2*i+1),torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1*pow(2,i-1),nStateConv1*pow(2,i),3).stride(1).padding(1))));
      stateConvLayers.push_back(register_module("stateConv2_"+std::to_string(2*i+2),torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1*pow(2,i),nStateConv1*pow(2,i),3).stride(1).padding(1))));
    }
  actionfc1 = register_module("actionfc1",torch::nn::Linear(4,nActionfc1));
  actionfc2 = register_module("actionfc2",torch::nn::Linear(nActionfc1,nActionfc2));
  actionfc3 = register_module("actionfc3",torch::nn::Linear(nActionfc2,FLAGS_sc1*pow(2,stateConvLayers.size()+2)));
}

torch::Tensor WorldModelGWImpl::encoderForward(torch::Tensor x)
{
  outputCopies = std::vector<torch::Tensor>();
  for (int i=0;i<nUnetLayers;i++)
    {
      x = stateConvLayers[2*i]->forward(x);
      x = torch::relu(x);
      x = stateConvLayers[2*i+1]->forward(x);
      x = torch::relu(x);
      std::cout<<x<<std::endl;
      x = torch::max_pool2d(x,2);
      std::cout<<x<<std::endl;
      outputCopies.push_back(x); //Making a backup that will be used in the decoder
    }
  return x;
}

torch::Tensor WorldModelGWImpl::actionForward(torch::Tensor x)
{
  //One-Hot encoding the batch of actions
  torch::Tensor y = torch::zeros({x.size(0),4});
  for (int i=0;i<x.size(0);i++)
    {
      y[i][*x[i].to(torch::Device(torch::kCPU)).data<int>()] = 1;
    }
  
  //Going through the MLP

  y = torch::relu(actionfc1->forward(y));  
  y = torch::relu(actionfc2->forward(y));
  y = torch::relu(actionfc3->forward(y));
  return y;
}
  

