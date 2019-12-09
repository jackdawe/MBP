#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H
#include "agent.h"
#include "../GridWorld/Models/convnetgw.h"
DECLARE_double(lr);
DECLARE_double(beta);
DECLARE_double(zeta);
DECLARE_int32(bs);
DECLARE_int32(n);
TORCH_MODULE(ConvNetGW);

template <class W, class M>
  class ActorCritic: public Agent<W>
{
 public:
  ActorCritic();
  ActorCritic(W world, M model);
  void evaluateRunValues();
  void backPropagate(torch::optim::Adam *opti);
  void train(int nEpisodes, float gamma, float beta, float zeta, float lr, int batchSize);
  void playOne();    
  void saveTrainingData();
  M getModel() const;

 private:  
  int episodeNumber;
  float gamma;
  float beta;
  float zeta;
  int batchSize;    
  torch::Tensor runStates;
  torch::Tensor runActions;
  vector<float> runRewards;
  vector<bool> runAreTerminal;
  torch::Tensor runValues;
  M model;
  
  vector<float> policyLossHistory;
  vector<float> valueLossHistory;
  vector<float> entropyHistory;
  vector<float> lossHistory;
  vector<float> vHistory;
};

#endif // ACTORCRITIC_H
