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
  torch::Tensor normalizeStates(torch::Tensor x, bool reverse=false);
  torch::Tensor normalizeDeltas(torch::Tensor x, bool reverse=false);  
  torch::Tensor normalizeActions(torch::Tensor x, bool reverse=false);
  torch::Tensor deltaToState(torch::Tensor stateBatch, torch::Tensor deltas);
  torch::Tensor moduloMSE(torch::Tensor target, torch::Tensor label, bool normalized=true);
  torch::Tensor penalityMSE(torch::Tensor target, torch::Tensor label, float valToPenalize, float weight);
  void generateSeed(int nTimesteps, int nRollouts, string filename);
  void generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp=0.1);
  void transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit, bool disp);
  void displayTAccuracy(int dataSetSize);
  void rewardAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit, bool disp);
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
