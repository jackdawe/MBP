#include "forwardss.h"

ForwardSSImpl::ForwardSSImpl(){}

ForwardSSImpl::ForwardSSImpl(int size, int nfc, int depth):
  size(size), nfc(nfc), depth(depth), savedStates(torch::zeros(0).to(usedDevice))
{
  init();
}

ForwardSSImpl::ForwardSSImpl(std::string filename):
  savedStates(torch::zeros(0).to(usedDevice))
{
  loadParams(filename);
}

void ForwardSSImpl::init()
{
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
  decoderLayers.push_back(register_module("POSITION OUT",torch::nn::Linear(nfc,2)));
  decoderLayers.push_back(register_module("VELOCITY OUT",torch::nn::Linear(nfc,2)));  

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
  for (unsigned int i=0;i<decoderLayers.size()-2;i++)
    {
      x = torch::prelu(decoderLayers[i]->forward(x),torch::full({1},0.001).to(usedDevice));
    }
  torch::Tensor posOut = decoderLayers[decoderLayers.size()-2]->forward(x);
  //  posOut = torch::sigmoid(posOut);
  torch::Tensor veloOut = decoderLayers.back()->forward(x);
  veloOut = torch::tanh(veloOut);
  return torch::cat({posOut,veloOut},1);
}

torch::Tensor ForwardSSImpl::rewardDecoderForward(torch::Tensor x)
{
  for (unsigned int i=0;i<rewardLayers.size()-1;i++)
    {
      x = torch::prelu(rewardLayers[i]->forward(x),torch::full({1},0.001).to(usedDevice));
    }
  return torch::tanh(rewardLayers.back()->forward(x));
}

void ForwardSSImpl::forward(torch::Tensor stateBatch, torch::Tensor actionBatch)
{
  stateBatch = stateBatch.to(usedDevice), actionBatch = actionBatch.to(usedDevice);    

  //Forward Pass

  torch::Tensor seOut = stateEncoderForward(ToolsSS().normalizeStates(stateBatch));
  torch::Tensor aeOut = actionEncoderForward(actionBatch);
  torch::Tensor x = torch::cat({seOut,aeOut},1);
  predictedState = stateDecoderForward(x);
  predictedReward = rewardDecoderForward(x).squeeze();

  //Rebuilding state from deltas

  predictedState = ToolsSS().normalizeDeltas(predictedState,true);
  predictedState = ToolsSS().deltaToState(stateBatch,predictedState);  
}

void ForwardSSImpl::computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels)
{
  //  cout<<torch::cat({predictedState.slice(1,0,4,1),stateLabels},1)[0].unsqueeze(0)<<endl;
  predictedState = ToolsSS().normalizeStates(predictedState);
  stateLabels = ToolsSS().normalizeStates(stateLabels);
  stateLoss = ToolsSS().moduloMSE(predictedState.slice(1,0,2,1),stateLabels.slice(1,0,2,1))+
  torch::mse_loss(predictedState.slice(1,2,4,1),stateLabels.slice(1,2,4,1));
  rewardLoss = torch::mse_loss(predictedReward,rewardLabels);
  //cout<<torch::cat({predictedState.slice(2,0,4,1),stateLabels.slice(2,0,4,1)},2)[0]<<endl;
  //  cout<<stateLabels[1000000000000000]<<endl;
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



