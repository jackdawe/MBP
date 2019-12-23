#include "forwardgw.h"

ForwardGWImpl::ForwardGWImpl():
  usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for ForwardGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout << "CUDA not available for ForwardGW: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
}

ForwardGWImpl::ForwardGWImpl(int size, int nConv1):
  size(size), nConv1(nConv1), usedDevice(torch::Device(torch::kCPU)), nUnetLayers(-1+log(size)/log(2))
{
  init();
}

ForwardGWImpl::ForwardGWImpl(std::string filename):
  usedDevice(torch::Device(torch::kCPU))
{
  loadParams(filename);
}

void ForwardGWImpl::init()
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for ForwardGW: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout << "CUDA not available for ForwardGW: training and inference will be done using CPU." << std::endl;
    }
    this->to(usedDevice);

    //Adding the convolutionnal layers of the encoder 

    stateConvLayers1.push_back(register_module("Encoder Conv1_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,nConv1,3).stride(1).padding(1))));
    stateConvLayers1.push_back(register_module("Encoder Conv1_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1,nConv1,3).stride(1).padding(1))));

    for (int i=1;i<nUnetLayers;i++)
      {
	stateConvLayers1.push_back(register_module("Encoder Conv"+std::to_string(i+1)+"_1",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1*pow(2,i-1),nConv1*pow(2,i),3).stride(1).padding(1))));
	stateConvLayers1.push_back(register_module("Encoder Conv"+std::to_string(i+1)+"_2",torch::nn::Conv2d(torch::nn::Conv2dOptions(nConv1*pow(2,i),nConv1*pow(2,i),3).stride(1).padding(1))));
      }

    //Adding the fully connected layers of the action MLP
    
    int fcOutputSize = nConv1*pow(2,nUnetLayers+2);
    actionfc1 = register_module("actionfc1",torch::nn::Linear(4,32));
    actionfc2 = register_module("actionfc2",torch::nn::Linear(32,64));
    actionfc3 = register_module("actionfc3",torch::nn::Linear(64,fcOutputSize));
    
    //Adding the transposed convolutionnal layers of the decoder
    
    nc_actEmb = fcOutputSize/4; //Number of channels of the action embedding after reshaping 
    nc_encoderOut = nConv1*pow(2,nUnetLayers-1); //Number of channels of the encoder output
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

    //Adding the convolutionnal layers of the reward CNN

    rewardConvLayers.push_back(register_module("Reward Conv1",torch::nn::Conv2d(torch::nn::Conv2dOptions(3,16,3).stride(1).padding(1))));
    for (int i=1;i<nUnetLayers;i++)
      {
	rewardConvLayers.push_back(register_module("Reward Conv"+std::to_string(i+1),torch::nn::Conv2d(torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1))));
      }
    rewardfcLayer = register_module("Reward FC",torch::nn::Linear(64,128));
    rewardOut = register_module("Reward Out",torch::nn::Linear(128,1));
}

torch::Tensor ForwardGWImpl::encoderForward(torch::Tensor x)
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

torch::Tensor ForwardGWImpl::actionForward(torch::Tensor x)
{
  x = torch::relu(actionfc1->forward(x));  
  x = torch::relu(actionfc2->forward(x));
  x = torch::relu(actionfc3->forward(x));
  return x;
}

torch::Tensor ForwardGWImpl::decoderForward(torch::Tensor x)
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
  x = torch::tanh(x);  
  return x;
}

torch::Tensor ForwardGWImpl::rewardForward(torch::Tensor x)
{
  for (int i=0;i<nUnetLayers;i++)
    {
      x = rewardConvLayers[i]->forward(x);
      x = torch::relu(x);
      x = torch::max_pool2d(x,2);
    }
  x = x.view({-1,64});
  x = rewardfcLayer->forward(x);
  x = torch::relu(x);
  x = rewardOut->forward(x);
  x = torch::tanh(x);
  return x;
}


void ForwardGWImpl::forward(torch::Tensor stateBatch, torch::Tensor actionBatch)
{
  stateBatch = stateBatch.to(usedDevice), actionBatch = actionBatch.to(usedDevice);

  /*PREDICTING THE NEXT STATE*/
  
  //Conversion to image if input is a batch of state vector
  bool imState = stateBatch.size(1)==3;        
  torch::Tensor x = stateBatch.clone();
  torch::Tensor y = x.clone();
  if (!imState)
    {
      x = ToolsGW().toRGBTensor(stateBatch.to(torch::Device(torch::kCPU))).to(usedDevice);            
    }

   vector<torch::Tensor> channels = torch::split(x,1,1);
   
   
  //Forward Pass

  torch::Tensor encoderOut = this->encoderForward(x);
  torch::Tensor actionEmbedding = this->actionForward(actionBatch);
  x = actionEmbedding.reshape({actionEmbedding.size(0),nc_actEmb,2,2});
  x = torch::cat({encoderOut,x},1);
  x = decoderForward(x);
  
  //Converting output into state vector if needed
  
  if(imState)
    {      
      predictedState = torch::cat({x,channels[1],channels[2]},1);   //Reconstituting the 3-channel image from the predicted state;
    }
  else
    {
      for (unsigned int s=0;s<stateBatch.size(0);s++)
	{
	  int agentPos = *torch::argmax(torch::round(x[s])).to(torch::Device(torch::kCPU)).data<long>();
	  int xmax = agentPos/size;
	  int ymax = agentPos%size;	  
	  y[s][0] = xmax;
	  y[s][1] = ymax;
	}
      predictedState = y;
    }

  /*PREDICTING THE REWARD ASSOCIATED TO THE TRANSITION*/

  x=torch::cat({x,channels[1],channels[2]},1);   //Reconstituting the 3-channel image from the predicted state
  predictedReward = rewardForward(x).squeeze();  
}

void ForwardGWImpl::computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels)
{
  stateLoss = torch::mse_loss(predictedState, stateLabels);
  rewardLoss = torch::mse_loss(predictedReward, rewardLabels);
}

void ForwardGWImpl::saveParams(std::string filename)
{
  std::ofstream f(filename);
  {
    f<<"###PARAMETERS FOR LOADING A FORWARD MODEL###"<<std::endl;
    f<<std::to_string(size)<<std::endl;
    f<<std::to_string(nConv1)<<std::endl;
  }
}

void ForwardGWImpl::loadParams(std::string filename)
{
  std::ifstream f(filename);
  if (!f)
    {
      std::cout<<"An error has occured while trying to load the ForwardGW model." << std::endl;
    }
  else {
    std::string line;
    std::getline(f,line);
    std::getline(f,line); size=stoi(line);
    std::getline(f,line); nConv1=stoi(line);
    nUnetLayers = -1+log(size)/log(2);
    init();
  }
}

torch::Device ForwardGWImpl::getUsedDevice()
{
  return usedDevice;
}

