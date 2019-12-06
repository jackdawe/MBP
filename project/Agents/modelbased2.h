#ifndef MODELBASED2_H
#define MODELBASED2_H
#include "../agent.h"
#include "../GridWorld/gridworld.h"
#include "../GridWorld/Models/forwardgw.h"
#include "../GridWorld/Models/plannergw.h"
TORCH_MODULE(ForwardGW);

template<class W, class F, class P>
  class ModelBased2: public Agent<W>
{
 public:  
  ModelBased2();
  ModelBased2(W world, F forwardModel); 
  ModelBased2(W world, F forwardModel, P planner);
  void learnForwardModel(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize=32, float lr=0.001);
  void gradientBasedPlanner(int nRollouts, int nTimesteps, int nGradientSteps, float lr);
  void saveTrainingData();
  F getForwardModel();
  vector<float> tensorToVector(torch::Tensor stateVector);
  
 private:
  F forwardModel;
  P planner;

  torch::Tensor actionSequence;
  torch::Tensor trajectory;
  torch::Device device;
  
  vector<float> rLossHistory;
  vector<float> sLossHistory;
};

#endif //MODELBASED2_H
