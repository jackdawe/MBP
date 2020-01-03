#ifndef FORWARDSS_H
#define FORWARDSS_H
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <math.h>
#include <fstream>
#include <iostream>
#include "../toolsss.h"

class ForwardSSImpl: public torch::nn::Module
{
 public:
  ForwardSSImpl();
  ForwardSSImpl(int size, int nfc, int depth);
  ForwardSSImpl(std::string filename);
  torch::Tensor stateEncoderForward(torch::Tensor x);
  torch::Tensor actionEncoderForward(torch::Tensor x);
  torch::Tensor stateDecoderForward(torch::Tensor x);
  torch::Tensor rewardDecoderForward(torch::Tensor x);
  void forward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool unnormalize=false);
  void computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels, bool normalize=true);
  
  void saveParams(std::string filename);
  void loadParams(std::string filename);
  
  torch::Device getUsedDevice();

  torch::Tensor predictedState;
  torch::Tensor predictedReward;
  torch::Tensor stateLoss;
  torch::Tensor rewardLoss;  
  
 private:
  void init();
  
  torch::Device usedDevice;
  int size;
  int nfc;
  int depth;
  
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> encoderLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> actionLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> decoderLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> rewardLayers;
};

#endif
