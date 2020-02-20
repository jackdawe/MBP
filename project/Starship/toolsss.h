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
  vector<float> tensorToVector(torch::Tensor stateVector);
  torch::Tensor normalizeStates(torch::Tensor x, bool reverse=false);
  torch::Tensor normalizeDeltas(torch::Tensor x, bool reverse=false);  
  torch::Tensor normalizeActions(torch::Tensor x, bool reverse=false);
  torch::Tensor deltaToState(torch::Tensor stateBatch, torch::Tensor deltas);
  torch::Tensor moduloMSE(torch::Tensor target, torch::Tensor label, bool normalized=true);
  torch::Tensor penalityMSE(torch::Tensor target, torch::Tensor label, float valToPenalize, float weight);
  void generateSeed(int nTimesteps, int nRollouts, string filename);
  torch::Tensor generateActions(int n, int nTimesteps, int distribution,float alpha, float std);
  void generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp=0.1, int aDist=0, float alhpa=1, float std=1);
  float comparePosMSE(torch::Tensor initState, int nWaypoints, torch::Tensor actionSequence, torch::Tensor estimate);
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
