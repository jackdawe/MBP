#ifndef TRANSITIONGW_H
#define TRANSITIONGW_H
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <math.h>

class TransitionGWImpl: public torch::nn::Module
{
 public:
  TransitionGWImpl();
  TransitionGWImpl(int size, int nStateConv1, int nActionfc1, int nActionfc2);
  torch::Tensor encoderForward(torch::Tensor x);
  torch::Tensor actionForward(torch::Tensor x);
  torch::Tensor decoderForward(torch::Tensor x);
  torch::Tensor rewardForward(torch::Tensor x);
  torch::Tensor predictState(torch::Tensor stateBatch, torch::Tensor actionBatch);

  torch::Device getUsedDevice();

 private:
  torch::Device usedDevice;

  std::vector<torch::Tensor> outputCopies;

  int nUnetLayers;
  int nc_actEmb;
  int nc_encoderOut;
  int nc_decoderIn;
  int nc_decoderConv1In;

  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> stateConvLayers1;
  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> stateConvLayers2;
  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> stateDeconvLayers;
  std::shared_ptr<torch::nn::LinearImpl> actionfc1;
  std::shared_ptr<torch::nn::LinearImpl> actionfc2;
  std::shared_ptr<torch::nn::LinearImpl> actionfc3;  
};

#endif
