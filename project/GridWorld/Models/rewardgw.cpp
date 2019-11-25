#include "rewardgw.h"
DEFINE_int32(sc1,16,"'State Conv 1': The number of feature maps in the first layer of the Unet encoder. Layer 2 will have twice as many feature maps and so on.");
DEFINE_int32(afc1,32,"'Action Fully Connected 1': The number of hidden units in the first layer of the MLP that maps action vectors to action vector embeddings");
DEFINE_int32(afc2,64,"'Action Fully Connected 2': The number of hidden units in the second layer of the MLP that maps action vectors to action vector embeddings");
DEFINE_int32(rc1,16,"'Reward Conv 1': the number of feature maps in the first layer of the ConvNet that maps predicted states to rewards. The next layer will have twice as many feature maps and so on. THe number of layers depend on the size of the map.");
DEFINE_int32(rfc,64,"'Reward Fully Connected': the number of hidden units in the fully connected layer of the ConvNet that maps predicted states to predicted rewards.");

RewardGWImpl::RewardGWImpl():
  usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for RewardGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout << "CUDA not available for RewardGW: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
}

RewardGWImpl::RewardGWImpl(int size,int nConv1, int nActionfc1, int nActionfc2):
  usedDevice(torch::Device(torch::kCPU)),nLayers(-1+log(size)/log(2)) 
{

  //Switching to CUDA if available
  
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for RewardGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout << "CUDA not available for RewardGW: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);

  //Adding the convolutionnal layers of the state CNN 

  convLayers.push_back(register_module("Conv 1_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nConv1,3).stride(1).padding(1))));
  convLayers.push_back(register_module("Conv 1_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1,nConv1,3).stride(1).padding(1))));

  for (int i=1;i<nLayers;i++)
    {
      convLayers.push_back(register_module("Conv "+std::to_string(i+1)+"_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1*pow(2,i-1),nConv1*pow(2,i),3).stride(1).padding(1))));
      convLayers.push_back(register_module("Conv "+std::to_string(i+1)+"_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1*pow(2,i),nConv1*pow(2,i),3).stride(1).padding(1))));
    }

  //Adding the fully connected layers of the action MLP

  nActionOut = nConv1*pow(2,nLayers+1);
  actionfc1 = register_module("Action FC 1",torch::nn::Linear(4,nActionfc1));
  actionfc2 = register_module("Action FC 2",torch::nn::Linear(nActionfc1,nActionfc2));
  actionfc3 = register_module("Action OUT",torch::nn::Linear(nActionfc2,nActionOut));
  
  //Building the output MLP 

  int nUnitsCount = nActionOut*2;

  for (int i=0;i<nLayers;i++)
    {
      rewardFCLayers.push_back(register_module("Reward FC " + std::to_string(i+1),torch::nn::Linear(nUnitsCount,nUnitsCount/2)));
      nUnitsCount/=2;
    }
  rewardFCLayers.push_back(register_module("Reward FC OUT", torch::nn::Linear(nUnitsCount,3)));
}

torch::Tensor RewardGWImpl::cnnForward(torch::Tensor x)
{
  for (int i=0;i<nLayers;i++)
    {
      x = convLayers[2*i]->forward(x);
      x = torch::relu(x);
      x = convLayers[2*i+1]->forward(x);
      x = torch::relu(x);
      x = torch::max_pool2d(x,2);
    }
  return x;
}

torch::Tensor RewardGWImpl::actionForward(torch::Tensor x)
{
  //One-Hot encoding the batch of actions

  torch::Tensor y = torch::zeros({x.size(0),4}).to(usedDevice);
  x = x.to(torch::kInt32);
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

torch::Tensor RewardGWImpl::rewardForward(torch::Tensor x)
{
  for (unsigned int i=0;i<rewardFCLayers.size()-1;i++)
    {
      x = torch::relu(rewardFCLayers[i]->forward(x));
    }
  x = rewardFCLayers[rewardFCLayers.size()-1]->forward(x);
  x = torch::log_softmax(x,1);
  return x;
}

torch::Tensor RewardGWImpl::predictReward(torch::Tensor stateBatch, torch::Tensor actionBatch)
{
  torch::Tensor cnnOut = cnnForward(stateBatch);
  cnnOut = cnnOut.view({-1,cnnOut.size(1)*cnnOut.size(2)*cnnOut.size(2)});
  torch::Tensor actionEmbedding = actionForward(actionBatch);
  
  torch::Tensor x = torch::cat({cnnOut,actionEmbedding},1);
  return rewardForward(x); 
}

torch::Device RewardGWImpl::getUsedDevice()
{
  return usedDevice;
}





