#ifndef MODELBASED_H
#define MODELBASED_H
#include "../agent.h"
#include "../GridWorld/gridworld.h"
#include "../GridWorld/Models/forwardgw.h"
#include "../GridWorld/Models/plannergw.h"
TORCH_MODULE(ForwardGW);
TORCH_MODULE(PlannerGW);

template<class W, class F, class P>
  class ModelBased: public Agent<W>
{
 public:  
  ModelBased();
  ModelBased(W world, F forwardModel); 
  ModelBased(W world, F forwardModel, P planner);
  void learnForwardModel(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize=32, float lr=0.001);
  void gradientBasedPlanner(int nRollouts, int nTimesteps, int nGradientSteps, float lr);
  void trainPolicyNetwork(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int n, int epochs, int batchSize=32, float lr=0.001);
  void playOne(int nRollouts, int nTimesteps, int nGradientSteps, float lr);
  void saveTrainingData();
  F getForwardModel();
  vector<float> tensorToVector(torch::Tensor stateVector);
  
 private:
  F forwardModel;
  P planner;

  torch::Tensor actionSequence;
  torch::Tensor trajectory;
  torch::Tensor reward;
  torch::Device device;
  
  vector<float> rLossHistory;
  vector<float> sLossHistory;
};

#endif //MODELBASED2_H
