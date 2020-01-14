#ifndef FORWARD_H
#define FORWARD_H
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <math.h>
#include <fstream>
#include <iostream>


class ForwardImpl: public torch::nn::Module
{
 public:
  ForwardImpl();
  virtual void forward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool restore=false);
  virtual void computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels);  
  
  torch::Tensor predictedState;
  torch::Tensor predictedReward;
  torch::Tensor stateLoss;
  torch::Tensor rewardLoss;  
  torch::Device usedDevice;
};

#endif
