#ifndef MODELBASED_H
#define MODELBASED_H
#include "../agent.h"
#include "../GridWorld/gridworld.h"
#include "../GridWorld/Models/transitiongw.h"
#include "../GridWorld/Models/rewardgw.h"
#include "../GridWorld/Models/plannergw.h"
TORCH_MODULE(TransitionGW);
TORCH_MODULE(RewardGW);
DECLARE_int32(K);
DECLARE_int32(T);
DECLARE_int32(gs);

template<class W, class T, class R, class P>
  class ModelBased: public Agent<W>
{
 public:  
  ModelBased();
  ModelBased(W world, T transitionFunction); //For learning the transition function
  ModelBased(W world, R rewardFunction); //For learning the reward function 
  ModelBased(W world, T transitionFunction, R rewardFunction, P planner);
  void learnTransitionFunction(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor labels, int epochs, int batchSize=32, float lr=0.001);
  void learnRewardFunction(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor labels, int epochs, int batchSize=32, float lr=0.001);
  void gradientBasedPlanner(int nRollouts, int nTimesteps, int nGradientSteps, float lr);
  void saveTrainingData();
  T getTransitionFunction();
  R getRewardFunction();
  vector<float> tensorToVector(torch::Tensor stateVector);
  
 private:
  T transitionFunction;
  R rewardFunction;
  P planner;

  torch::Tensor actionSequence;
  torch::Tensor trajectory;

  vector<float> rLossHistory;
  vector<float> sLossHistory;
};

#endif //MODELBASED_H
