#ifndef TOOLSGW_H
#define TOOLSGW_H
#include "gridworld.h"

class ToolsGW
{
 public:
  ToolsGW();
  ToolsGW(GridWorld gw);
  vector<float> tensorToVector(torch::Tensor stateVector);
  torch::Tensor toRGBTensor(torch::Tensor batch);
  torch::Tensor generateActions(int n, int nTimesteps);
  void generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp=0.1);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit);
  void displayTAccuracy(int dataSetSize);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit);
  void displayRAccuracy();
  torch::Tensor tMSE;
  torch::Tensor rMSE;
 private:
  GridWorld gw;
  int tScores;
  vector<int> rScores;
  vector<int> rCounts;

};

#endif 
