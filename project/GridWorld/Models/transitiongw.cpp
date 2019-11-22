#include "transitiongw.h"

TransitionGWImpl::TransitionGWImpl():
  usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for TransitionGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout <<"Training and inference will be done using CPU for TransitionGW"<<std::endl;
      std::cout << "CUDA not available for TransitionGW: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
}

TransitionGWImpl::TransitionGWImpl(int size, int nStateConv1, int nActionfc1, int nActionfc2):
  usedDevice(torch::Device(torch::kCPU)), nUnetLayers(-1+log(size)/log(2))
{
    if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for TransitionGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout << "CUDA not available for TransitionGW: training and inference will be done using CPU." << std::endl;
    }
    this->to(usedDevice);

    //Adding the convolutionnal layers of the encoder 

    stateConvLayers1.push_back(register_module("Encoder Conv1_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nStateConv1,3).stride(1).padding(1))));
    stateConvLayers1.push_back(register_module("Encoder Conv1_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1,nStateConv1,3).stride(1).padding(1))));

    for (int i=1;i<nUnetLayers;i++)
      {
	stateConvLayers1.push_back(register_module("Encoder Conv"+std::to_string(i+1)+"_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1*pow(2,i-1),nStateConv1*pow(2,i),3).stride(1).padding(1))));
	stateConvLayers1.push_back(register_module("Encoder Conv"+std::to_string(i+1)+"_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nStateConv1*pow(2,i),nStateConv1*pow(2,i),3).stride(1).padding(1))));
      }

    //Adding the fully connected layers of the action MLP
    
    int fcOutputSize = nStateConv1*pow(2,nUnetLayers+2);
    actionfc1 = register_module("actionfc1",torch::nn::Linear(4,nActionfc1));
    actionfc2 = register_module("actionfc2",torch::nn::Linear(nActionfc1,nActionfc2));
    actionfc3 = register_module("actionfc3",torch::nn::Linear(nActionfc2,fcOutputSize));
    
    //Adding the transposed convolutionnal layers of the decoder
    
    nc_actEmb = fcOutputSize/4; //Number of channels of the action embedding after reshaping 
    nc_encoderOut = nStateConv1*pow(2,nUnetLayers-1); //Number of channels of the encoder output
    nc_decoderIn = nc_actEmb + nc_encoderOut; //Number of channels after concatenation with the action embedding 
    nc_decoderConv1In = (nc_encoderOut + nc_decoderIn)/2; //Number of channels after the first transposed convolution and concatenation, before the first convolutionnal layer of the decoder
    
    int chanCount = nc_decoderIn;
    stateDeconvLayers.push_back(register_module("Decoder Deconv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(chanCount,chanCount/2.,3).stride(2).dilation(1).padding(1).output_padding(1).transposed(true))));
    chanCount = nc_encoderOut;
    for (int i=1;i<nUnetLayers;i++)
      {
	chanCount/=2.;
	stateDeconvLayers.push_back(register_module("Decoder Deconv" + std::to_string(i+1),torch::nn::Conv2d(torch::nn::Conv2dOptions(chanCount,chanCount/2,3).stride(2).dilation(1).padding(1).output_padding(1).transposed(true))));
      }

    //Adding the convolutionnal layers of the decoder
    
    chanCount = nc_decoderConv1In;
    stateConvLayers2.push_back(register_module("Decoder Conv1_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(nc_decoderConv1In,nc_encoderOut/2,3).stride(1).padding(1))));
    stateConvLayers2.push_back(register_module("Decoder Conv1_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nc_encoderOut/2,nc_encoderOut/2,3).stride(1).padding(1))));
    chanCount = nc_encoderOut/2;
    for (int i=1;i<nUnetLayers-1;i++)
      {
	stateConvLayers2.push_back(register_module("Decoder Conv"+std::to_string(i+1)+"_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(chanCount,chanCount/2,3).stride(1).padding(1))));
	chanCount/=2.;
	stateConvLayers2.push_back(register_module("Decoder Conv"+std::to_string(i+1)+"_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(chanCount,chanCount,3).stride(1).padding(1))));
      }
    
    stateConvLayers2.push_back(register_module("Decoder Conv Final",torch::nn::Conv2d(torch::nn::Conv2dOptions(chanCount/2,1,3).stride(1).padding(1))));
}

torch::Tensor TransitionGWImpl::encoderForward(torch::Tensor x)
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

torch::Tensor TransitionGWImpl::actionForward(torch::Tensor x)
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

torch::Tensor TransitionGWImpl::decoderForward(torch::Tensor x)
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
  int imSize = x.size(2);
  x = x.reshape({x.size(0),imSize*imSize});
  x = torch::softmax(x,1);
  x = x.reshape({x.size(0),imSize,imSize});
  return x;
}

torch::Tensor TransitionGWImpl::predictState(torch::Tensor stateBatch, torch::Tensor actionBatch)
{
  torch::Tensor encoderOut = this->encoderForward(stateBatch);
  torch::Tensor actionEmbedding = this->actionForward(actionBatch);
  torch::Tensor x = actionEmbedding.reshape({actionEmbedding.size(0),nc_actEmb,2,2});
  x = torch::cat({encoderOut,x},1);
  x = decoderForward(x);
  return x;
}

torch::Device TransitionGWImpl::getUsedDevice()
{
  return usedDevice;
}

