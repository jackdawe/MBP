#ifndef MODELBASED_H
#define MODELBASED_H
#include "../agent.h"
#include "../GridWorld/gridworld.h"
#include "../GridWorld/Models/forwardgw.h"
#include "../GridWorld/Models/plannergw.h"
#include "../Starship/spaceworld.h"
#include "../Starship/Models/forwardss.h"
#include "../Starship/toolsss.h"
TORCH_MODULE(ForwardGW);
TORCH_MODULE(PlannerGW);
TORCH_MODULE(ForwardSS);

template<class W, class F, class P>
  class ModelBased: public Agent<W>
{
 public:  
  ModelBased();
  ModelBased(W world, F forwardModel); 
  ModelBased(W world, F forwardModel, P planner);
  void learnForwardModel(torch::optim::Adam *optimizer, torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize=32, float beta=1, bool allStatesProvided = true);
  void gradientBasedPlanner(torch::Tensor initState, ActionSpace actionSpace, int nRollouts, int nTimesteps, int nGradientSteps, float lr, torch::Tensor initActions=torch::zeros(0));
  void trainPolicyNetwork(torch::Tensor actionInputs, torch::Tensor stateInputs, int epochs, int batchSize=32, float lr=0.001);
  void playOne(torch::Tensor initState, ActionSpace actionSpace, int nRollouts, int nTimesteps, int nGradientSteps, float lr, torch::Tensor initActions=torch::zeros(0));
  void saveTrainingData();
  F getForwardModel();
  torch::Tensor splitDim(torch::Tensor x, int bs, int T);
  torch::Tensor mergeDim(torch::Tensor x);

  //GBP methods

  torch::Tensor getInitTokens(vector<DiscreteAction> discreteActions, int T, int K);
  torch::Tensor getInitCA(int nca, int T, int K);  
  
  vector<torch::Tensor> breakDAIntoTokens(vector<DiscreteAction> discreteActions, torch::Tensor toOptiDA);
  torch::Tensor tokensToOneHot(vector<torch::Tensor> daTokens, int daSize, int t);

  torch::Tensor getFinalDA(vector<DiscreteAction> discreteActions, torch::Tensor optiDA);
  torch::Tensor getFinalCA(vector<ContinuousAction> continuousActions, torch::Tensor optiCA);
  float computeTrueReward(torch::Tensor initState, vector<DiscreteAction> discreteActions, torch::Tensor actions, vector<ContinuousAction> continuousActions, int nca, int nda, int nTimesteps);
  
 private:
  F forwardModel;
  P planner;
  torch::Tensor actionSequence;
  torch::Tensor trajectory;
  torch::Tensor reward;
  torch::Device device;
  
  vector<float> rLossHistory;
  vector<float> sLossHistory;
  vector<float> pLossHistory;
};

#endif //MODELBASED2_H
