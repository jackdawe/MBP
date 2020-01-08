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
  torch::Tensor deltaToState(torch::Tensor stateBatch, torch::Tensor deltas);
  void generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp=0.1, float edgeSpawnProp=0.1);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit);
  void displayTAccuracy(int dataSetSize);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit);
  void displayRAccuracy();
  torch::Tensor pMSE;
  torch::Tensor vMSE;
  torch::Tensor rMSE;
 private:
  SpaceWorld sw;
  vector<int> tScores;
  vector<int> rScores;
  vector<int> rCounts;
};

#endif 
