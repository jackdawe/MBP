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

  encoderLayers.push_back(register_module("State Encoder FC1",torch::nn::Linear(size,nfc)));
  for (int i=1;i<depth;i++)
    {
      encoderLayers.push_back(register_module("State Encoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }

  //Adding the layers of the action encoder

  actionLayers.push_back(register_module("Action Encoder FC1",torch::nn::Linear(6,nfc)));
  for (int i=1;i<depth;i++)
    {
      actionLayers.push_back(register_module("Action Encoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }

    //Adding the layers of the state decoder

  decoderLayers.push_back(register_module("State Decoder FC1",torch::nn::Linear(2*nfc,nfc)));
  for (int i=1;i<depth;i++)
    {
      decoderLayers.push_back(register_module("State Decoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }
  decoderLayers.push_back(register_module("State Decoder OUT",torch::nn::Linear(nfc,4)));

    //Adding the layers of the reward decoder

  rewardLayers.push_back(register_module("Reward Decoder FC1",torch::nn::Linear(2*nfc,nfc)));
  for (int i=1;i<depth;i++)
    {
      rewardLayers.push_back(register_module("Reward Decoder FC"+std::to_string(i+1),torch::nn::Linear(nfc,nfc)));
    }
  rewardLayers.push_back(register_module("Reward Decoder OUT",torch::nn::Linear(nfc,1)));
}

torch::Tensor ForwardSSImpl::actionEncoderForward(torch::Tensor x)
{
  for (int i=0;i<depth;i++)
    {
      x = torch::relu(actionLayers[i]->forward(x));
    }
  return x;
}

torch::Tensor ForwardSSImpl::stateEncoderForward(torch::Tensor x)
{
  for (int i=0;i<depth;i++)
    {
      x = torch::relu(encoderLayers[i]->forward(x));
    }
  return x;
}

torch::Tensor ForwardSSImpl::stateDecoderForward(torch::Tensor x)
{
  for (int i=0;i<depth;i++)
    {
      x = torch::relu(decoderLayers[i]->forward(x));
    }
  return torch::tanh(decoderLayers.back()->forward(x));
}

torch::Tensor ForwardSSImpl::rewardDecoderForward(torch::Tensor x)
{
  for (int i=0;i<depth;i++)
    {
      x = torch::relu(rewardLayers[i]->forward(x));
    }
  return torch::tanh(rewardLayers.back()->forward(x));
}

void ForwardSSImpl::forward(torch::Tensor stateBatch, torch::Tensor actionBatch)
{

  int n = stateBatch.size(0);

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
  predictedReward = rewardDecoderForward(x);
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

