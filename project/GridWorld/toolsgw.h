#ifndef TOOLSGW_H
#define TOOLSGW_H
#include "gridworld.h"
DECLARE_double(wp);
DECLARE_bool(wn);
DECLARE_double(sd);

class ToolsGW
{
 public:
  ToolsGW();
  ToolsGW(GridWorld gw);
  void generateDataSet(string path, int nmaps, int n,float winProp=0.1, bool noise=false, float sigma=0.25);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels);
 private:
  GridWorld gw;
};

#endif 
