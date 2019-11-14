#include "worldmodelgw.h"
DEFINE_int32(sc1,8,"'State Conv 1': The number of feature maps in the first layer of the Unet encoder. Layer 2 will have twice as many feature maps and so on.");
DEFINE_int32(afc1,32,"'Action Fully Connected 1': The number of hidden units in the first layer of the MLP that maps action vectors to action vector embeddings");
DEFINE_int32(afc2,64,"'Action Fully Connected 2': The number of hidden units in the second layer of the MLP that maps action vectors to action vector embeddings");
DEFINE_int32(rc1,8,"'Reward Conv 1': the number of feature maps in the first layer of the ConvNet that maps predicted states to rewards. The next layer will have twice as many feature maps and so on. THe number of layers depend on the size of the map.");
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

WorldModelGWImpl::WorldModelGWImpl(int size,int nStateConv1, int nActionfc1, int nActionfc2, int nRewardConv1, int nRewardfc):
  usedDevice(torch::Device(torch::kCPU)), size(size), nStateConv1(nStateConv1), nActionfc1(nActionfc1), nActionfc2(nActionfc2), nRewardfc(nRewardfc), nRewardConv1(nRewardConv1) 
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

  //Adding the convolutionnal layers of the encoder 

  stateConvLayers1.push_back(register_module("Encoder Conv1_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nStateConv1,3).stride(1).padding(1))));
  stateConvLayers1.push_back(register_module("Encoder Conv1_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1,nStateConv1,3).stride(1).padding(1))));

  for (int i=1;i<nUnetLayers;i++)
    {
      stateConvLayers1.push_back(register_module("Encoder Conv"+std::to_string(i+1)+"_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1*pow(2,i-1),nStateConv1*pow(2,i),3).stride(1).padding(1))));
      stateConvLayers1.push_back(register_module("Encoder Conv"+std::to_string(i+1)+"_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1*pow(2,i),nStateConv1*pow(2,i),3).stride(1).padding(1))));
    }

  //Adding the fully connected layers of the action MLP

  int fcOutputSize = FLAGS_sc1*pow(2,nUnetLayers+2);
  actionfc1 = register_module("actionfc1",torch::nn::Linear(4,nActionfc1));
  actionfc2 = register_module("actionfc2",torch::nn::Linear(nActionfc1,nActionfc2));
  actionfc3 = register_module("actionfc3",torch::nn::Linear(nActionfc2,fcOutputSize));

  //Adding the transposed convolutionnal layers of the decoder

  int deconvIn = 3.*fcOutputSize/8;
  stateDeconvLayers.push_back(register_module("Decoder Deconv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(deconvIn,deconvIn/2.,3).stride(2).dilation(1).padding(1).output_padding(1).transposed(true))));
  deconvIn = (2*deconvIn/3)/2;
  for (int i=1;i<nUnetLayers;i++)
    {
      deconvIn/=2.;
      stateDeconvLayers.push_back(register_module("Decoder Deconv" + std::to_string(i+1),torch::nn::Conv2d(torch::nn::Conv2dOptions(deconvIn,deconvIn/2,3).stride(2).dilation(1).padding(1).output_padding(1).transposed(true))));
    }
  
  //Adding the convolutionnal layers of the decoder
  int convIn = fcOutputSize/4.;
  stateConvLayers2.push_back(register_module("Decoder Conv1_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(convIn,convIn/4.,3).stride(1).padding(1))));
  convIn/=4.;
  stateConvLayers2.push_back(register_module("Decoder Conv1_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(convIn,convIn,3).stride(1).padding(1))));

  for (int i=1;i<nUnetLayers-1;i++)
    {
      stateConvLayers2.push_back(register_module("Decoder Conv"+std::to_string(i+1)+"_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(convIn,convIn/2,3).stride(1).padding(1))));
      convIn/=2.;
      stateConvLayers2.push_back(register_module("Decoder Conv"+std::to_string(i+1)+"_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(convIn,convIn,3).stride(1).padding(1))));
    }

  stateConvLayers2.push_back(register_module("Decoder Conv Final",torch::nn::Conv2d(torch::nn::Conv2dOptions(convIn/2,3,3).stride(1).padding(1))));

  //Building the CNN of the reward function

  rewardConvLayers.push_back(register_module("Reward CNN Conv"+std::to_string(1),torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nRewardConv1,3).stride(1).padding(1))));
  for (int i=1;i<nUnetLayers;i++)
    {
      rewardConvLayers.push_back(register_module("Reward CNN Conv"+std::to_string(i+1),torch::nn::Conv2d(torch::nn::Conv2dOptions(nRewardConv1*pow(2,i-1),nRewardConv1*pow(2,i),3).stride(1).padding(1))));
    }
  rewardfc = register_module("Reward CNN fc", torch::nn::Linear(4*nRewardConv1*pow(2,nUnetLayers-1),nRewardfc));
  rewardOut = register_module("Reward CNN out", torch::nn::Linear(nRewardfc,1));
}

torch::Tensor WorldModelGWImpl::encoderForward(torch::Tensor x)
{
  outputCopies = std::vector<torch::Tensor>();
  for (int i=0;i<nUnetLayers;i++)
    {
      x = stateConvLayers1[2*i]->forward(x);
      x = torch::relu(x);
      x = stateConvLayers1[2*i+1]->forward(x);
      x = torch::relu(x);
      x = torch::max_pool2d(x,2);
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

torch::Tensor WorldModelGWImpl::decoderForward(torch::Tensor x)
{
  for (int i=0;i<nUnetLayers-1;i++)
    {
      x = stateDeconvLayers[i]->forward(x);
      x = torch::cat({x,outputCopies[nUnetLayers-i-2]},1);
      x = stateConvLayers2[2*i]->forward(x);
      x = torch::relu(x);
      x = stateConvLayers2[2*i+1]->forward(x);
      x = torch::relu(x);
    }
  x = stateDeconvLayers[nUnetLayers-1]->forward(x);
  x = stateConvLayers2[2*(nUnetLayers-1)]->forward(x);
  x = torch::sigmoid(x);
  return x;
} 

torch::Tensor WorldModelGWImpl::rewardForward(torch::Tensor x)
{
  for (int i=0;i<nUnetLayers;i++)
    {
      x = rewardConvLayers[i]->forward(x);
      x = torch::relu(x);
      x = torch::max_pool2d(x,2);
    }
  x = x.view({-1,4*nRewardConv1*pow(2,nUnetLayers-1)});
  x = rewardfc->forward(x);
  x = rewardOut->forward(x);
  x = torch::tanh(x);
  return x;
}
  

