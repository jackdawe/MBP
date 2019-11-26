#ifndef REWARDGW_H
#define REWARDGW_H
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS
#include <gflags/gflags.h>
#include <math.h>
#include "../toolsgw.h"
DECLARE_int32(sc1);
DECLARE_int32(sc2);
DECLARE_int32(afc1);
DECLARE_int32(afc2);
DECLARE_int32(aout);
DECLARE_int32(rc1);
DECLARE_int32(rc2);
DECLARE_int32(rfc);

class RewardGWImpl: public torch::nn::Module
{
 public:
  RewardGWImpl();
  RewardGWImpl(int size,int nConv1, int nActionfc1, int nActionfc2);
  RewardGWImpl(std::string filename);
  torch::Tensor cnnForward(torch::Tensor x);
  torch::Tensor actionForward(torch::Tensor x);
  torch::Tensor rewardForward(torch::Tensor x);
  torch::Tensor predictReward(torch::Tensor stateBatch, torch::Tensor actionBatch);

  void saveParams(std::string filename);
  void loadParams(std::string filename);
  
  torch::Device getUsedDevice();
  
 private:
  void init();
  
  torch::Device usedDevice;

  int size;
  int nConv1;
  int nActionfc1;
  int nActionfc2;
  
  int nLayers;
  int nActionOut;

  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> convLayers;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> rewardFCLayers;
  std::shared_ptr<torch::nn::LinearImpl> actionfc1;
  std::shared_ptr<torch::nn::LinearImpl> actionfc2;
  std::shared_ptr<torch::nn::LinearImpl> actionfc3;  
};
#endif //WORLDMODELGW_H
