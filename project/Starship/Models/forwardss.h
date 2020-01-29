#ifndef FORWARDSS_H
#define FORWARDSS_H
#include "../toolsss.h"
#include "../../forward.h"

class ForwardSSImpl: public ForwardImpl
{
 public:
  ForwardSSImpl();
  ForwardSSImpl(int size, int nfc, int depth);
  ForwardSSImpl(std::string filename);
  torch::Tensor stateEncoderForward(torch::Tensor x);
  torch::Tensor actionEncoderForward(torch::Tensor x);
  torch::Tensor stateDecoderForward(torch::Tensor x);
  torch::Tensor rewardDecoderForward(torch::Tensor x);
  void forward(torch::Tensor stateBatch, torch::Tensor actionBatch);
  void computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels);
  
  void saveParams(std::string filename);
  void loadParams(std::string filename);
  
 private:
  void init();
  
  int size;
  int nfc;
  int depth;
  
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> encoderLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> actionLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> decoderLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> rewardLayers;
  torch::Tensor savedStates;  
};

#endif
