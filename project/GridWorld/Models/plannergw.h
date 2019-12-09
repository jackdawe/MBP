#ifndef PLANNERGW_H
#define PLANNERGW_H
#undef slots
#include "../toolsgw.h"
#include <torch/torch.h>
#define slots Q_SLOTS
#include <gflags/gflags.h>


class PlannerGWImpl: public torch::nn::Module
{
 public:
  PlannerGWImpl();
  PlannerGWImpl(int size,int nConv,int nfc);
  torch::Tensor forward(torch::Tensor stateBatch);

  void init();
  void saveParams(std::string filename);
  void loadParams(std::string filename);
  
  torch::Device getUsedDevice();
  
 private:
  torch::Device usedDevice;
  int size;
  int nConv;
  int nfc;
  int nLayers;

  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> convLayers;
  std::shared_ptr<torch::nn::LinearImpl> fc;
  std::shared_ptr<torch::nn::LinearImpl> out;
};

#endif //PLANNERGW_H
