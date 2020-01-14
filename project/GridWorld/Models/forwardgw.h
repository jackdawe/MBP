#ifndef FORWARDGW_H
#define FORWARDGW_H
#include "../../forward.h"
#include "../toolsgw.h"

class ForwardGWImpl: public ForwardImpl
{
 public:
  ForwardGWImpl();
  ForwardGWImpl(int size, int nConv1);
  ForwardGWImpl(std::string filename);
  torch::Tensor encoderForward(torch::Tensor x);
  torch::Tensor actionForward(torch::Tensor x);
  torch::Tensor decoderForward(torch::Tensor x);
  torch::Tensor rewardForward(torch::Tensor x);
  void forward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool unnormalize=false);
  void computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels);
  
  void saveParams(std::string filename);
  void loadParams(std::string filename);
  
 private:
  void init();
  
  std::vector<torch::Tensor> outputCopies;

  int size;
  int nConv1;
  
  int nUnetLayers;
  int nc_actEmb;
  int nc_encoderOut;
  int nc_decoderIn;
  int nc_decoderConv1In;

  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> stateConvLayers1;
  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> stateConvLayers2;
  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> stateDeconvLayers;
  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> rewardConvLayers;
  std::shared_ptr<torch::nn::LinearImpl> actionfc1;
  std::shared_ptr<torch::nn::LinearImpl> actionfc2;
  std::shared_ptr<torch::nn::LinearImpl> actionfc3;
  std::shared_ptr<torch::nn::LinearImpl> rewardfcLayer;
  std::shared_ptr<torch::nn::LinearImpl> rewardOut;
};

#endif
