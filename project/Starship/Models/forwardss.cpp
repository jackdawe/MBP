#include "forwardss.h"

ForwardSSImpl::ForwardSSImpl():
  usedDevice(torch::Device(torch::kCPU))
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for ForwardSS: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
    else
    {
      std::cout << "CUDA not available for ForwardSS: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);
}

ForwardSSImpl::ForwardSSImpl(int size, int nfc, int depth):
  size(size), nfc(nfc), usedDevice(torch::Device(torch::kCPU)), depth(depth)
{
  init();
}

ForwardSSImpl::ForwardSSImpl(std::string filename):
  usedDevice(torch::Device(torch::kCPU))
{
  loadParams(filename);
}

void ForwardSSImpl::init()
{
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA detected for ForwardSS: training and inference will be done using CUDA." << std::endl;
      usedDevice = torch::Device(torch::kCUDA);
    }
  else
    {
      std::cout << "CUDA not available for ForwardSS: training and inference will be done using CPU." << std::endl;
    }
  this->to(usedDevice);

  //Adding the layers of the state encoder

  encoderLayers.push_back(register_module("State Encoder IN",torch::nn::Linear(size,nfc)));
  for (int i=0;i<depth;i++)
    {
      encoderLayers.push_back(register_module("State Encoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }
  encoderLayers.push_back(register_module("State Encoder OUT",torch::nn::Linear(nfc,nfc)));

  //Adding the layers of the action encoder

  actionLayers.push_back(register_module("Action Encoder IN",torch::nn::Linear(6,nfc)));
  for (int i=0;i<depth;i++)
    {
      actionLayers.push_back(register_module("Action Encoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }
  actionLayers.push_back(register_module("Action Encoder OUT",torch::nn::Linear(nfc,nfc)));

    //Adding the layers of the state decoder

  decoderLayers.push_back(register_module("State Decoder IN",torch::nn::Linear(2*nfc,nfc)));
  for (int i=0;i<depth;i++)
    {
      decoderLayers.push_back(register_module("State Decoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }
  decoderLayers.push_back(register_module("State Decoder OUT",torch::nn::Linear(nfc,4)));

    //Adding the layers of the reward decoder

  rewardLayers.push_back(register_module("Reward Decoder IN",torch::nn::Linear(2*nfc,nfc)));
  for (int i=0;i<depth;i++)
    {
      rewardLayers.push_back(register_module("Reward Decoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }
  rewardLayers.push_back(register_module("Reward Decoder OUT",torch::nn::Linear(nfc,1)));
}

torch::Tensor ForwardSSImpl::actionEncoderForward(torch::Tensor x)
{
  for (unsigned int i=0;i<actionLayers.size();i++)
    {
      x = torch::prelu(actionLayers[i]->forward(x),torch::full({1},0.001).to(usedDevice));
    }
  return x;
}

torch::Tensor ForwardSSImpl::stateEncoderForward(torch::Tensor x)
{
  for (unsigned int i=0;i<encoderLayers.size();i++)
    {
      x = torch::prelu(encoderLayers[i]->forward(x),torch::full({1},0.001).to(usedDevice));
    }
  return x;
}

torch::Tensor ForwardSSImpl::stateDecoderForward(torch::Tensor x)
{
  for (unsigned int i=0;i<decoderLayers.size()-1;i++)
    {
      x = torch::prelu(decoderLayers[i]->forward(x),torch::full({1},0.001).to(usedDevice));
    }
  return decoderLayers.back()->forward(x);
}

torch::Tensor ForwardSSImpl::rewardDecoderForward(torch::Tensor x)
{
  for (unsigned int i=0;i<rewardLayers.size()-1;i++)
    {
      x = torch::prelu(rewardLayers[i]->forward(x),torch::full({1},0.001).to(usedDevice));
    }
  return rewardLayers.back()->forward(x);
}

void ForwardSSImpl::forward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool unnormalize)
{
  stateBatch = stateBatch.to(usedDevice), actionBatch = actionBatch.to(usedDevice);    
  stateBatch = ToolsSS().normalize(stateBatch);
	  
  //Splitting varying and constant parts of the state vector
  std::vector<torch::Tensor> split = torch::split(stateBatch,4,1);

  //Forward Pass

  torch::Tensor seOut = stateEncoderForward(stateBatch);
  torch::Tensor aeOut = actionEncoderForward(actionBatch);
  torch::Tensor x = torch::cat({seOut,aeOut},1);
  predictedState = stateDecoderForward(x);
  for (int i=1;i<split.size();i++)
    {
      predictedState = torch::cat({predictedState,split[i]},1);
    }
  if (unnormalize)
    {
      predictedState = ToolsSS().normalize(predictedState,true);
    }
  predictedReward = rewardDecoderForward(x).squeeze();
}

void ForwardSSImpl::computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels, bool normalize)
{  
  if (normalize)
    {
      int n = predictedState.size(0), T= predictedState.size(1), s = predictedState.size(2);
      stateLabels = ToolsSS().normalize(stateLabels.reshape({n*T,s})).reshape({n,T,s});
    }  
  std::vector<torch::Tensor> stateOutputsChunks = torch::split(predictedState,4,2);
  std::vector<torch::Tensor> slBatchChunks = torch::split(stateLabels,4,2);  
  stateLoss = torch::mse_loss(stateOutputsChunks[0],slBatchChunks[0])+torch::mse_loss(stateOutputsChunks[1],slBatchChunks[1]);
  rewardLoss = torch::mse_loss(predictedReward,rewardLabels);
}

void ForwardSSImpl::saveParams(std::string filename)
{
  std::ofstream f(filename);
  {
    f<<"###PARAMETERS FOR LOADING A FORWARD MODEL###"<<std::endl;
    f<<std::to_string(size)<<std::endl;
    f<<std::to_string(nfc)<<std::endl;
    f<<std::to_string(depth)<<std::endl;
  }
}

void ForwardSSImpl::loadParams(std::string filename)
{
  std::ifstream f(filename);
  if (!f)
    {
      std::cout<<"An error has occured while trying to load the ForwardSS model." << std::endl;
    }
  else {
    std::string line;
    std::getline(f,line);
    std::getline(f,line); size=stoi(line);
    std::getline(f,line); nfc=stoi(line);
    std::getline(f,line); depth=stoi(line);
    init();
  }
}

torch::Device ForwardSSImpl::getUsedDevice()
{
  return usedDevice;
}

