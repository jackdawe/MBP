#ifndef TOOLSGW_H
#define TOOLSGW_H
#include "gridworld.h"

class ToolsGW
{
 public:
  ToolsGW();
  ToolsGW(GridWorld gw);
  torch::Tensor toRGBTensor(torch::Tensor batch);
  //  cv::Mat toRGBMat(torch::Tensor stateTensor);
  void generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp=0.1);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels);
 private:
  GridWorld gw;
};

#endif 
