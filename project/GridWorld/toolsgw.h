#ifndef TOOLSGW_H
#define TOOLSGW_H
#include "gridworld.h"
DECLARE_double(wp);

class ToolsGW
{
 public:
  ToolsGW();
  ToolsGW(GridWorld gw);
  void generateDataSet(int n,float winProp=0.1);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels);
 private:
  GridWorld gw;
};

#endif 
