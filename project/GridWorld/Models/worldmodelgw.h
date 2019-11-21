#ifndef WORLDMODELGW_H
#define WORLDMODELGH_H
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <gflags/gflags.h>
#include <math.h>
DECLARE_int32(sc1);
DECLARE_int32(sc2);
DECLARE_int32(afc1);
DECLARE_int32(afc2);
DECLARE_int32(aout);
DECLARE_int32(rc1);
DECLARE_int32(rc2);
DECLARE_int32(rfc);

class WorldModelGWImpl: public torch::nn::Module
{
 public:
  WorldModelGWImpl();
  WorldModelGWImpl(int size, int nStateConv1, int nActionfc1, int nActionfc2, int nRewardConv1, int nRewardfc);
  torch::Tensor encoderForward(torch::Tensor x);
  torch::Tensor actionForward(torch::Tensor x);
  torch::Tensor decoderForward(torch::Tensor x);
  torch::Tensor rewardForward(torch::Tensor x);
  torch::Tensor predictState(torch::Tensor stateBatch, torch::Tensor actionBatch);
  torch::Tensor predictReward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool inputAvailable = false);

  torch::Device getUsedDevice();
  torch::Tensor getDecoderIn();
  torch::Tensor getDecoderOut();
  
 private:
  torch::Device usedDevice;
  int size;
  int nStateConv1;
  int nActionfc1;
  int nActionfc2;
  int nRewardConv1;
  int nRewardfc;

  std::vector<torch::Tensor> outputCopies;

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
  std::shared_ptr<torch::nn::LinearImpl> rewardfc1;
  std::shared_ptr<torch::nn::LinearImpl> rewardfc2;
  std::shared_ptr<torch::nn::LinearImpl> rewardOut;
  std::shared_ptr<torch::nn::LinearImpl> test;

  torch::Tensor decoderIn;
  torch::Tensor decoderOut;
};
#endif //WORLDMODELGW_H
