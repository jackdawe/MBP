#ifndef TOOLSSS_H
#define TOOLSSS_H
#include "spaceworld.h"
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS

class ToolsSS
{
 public:
  ToolsSS();
  ToolsSS(SpaceWorld sw);
  torch::Tensor normalize(torch::Tensor x, bool reverse=false);
  void generateDataSet(string path, int nmaps, int n, int nTimesteps, float winProp=0.1);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels);
 private:
  SpaceWorld sw;
};

#endif 
