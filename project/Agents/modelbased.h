#ifndef MODELBASED_H
#define MODELBASED_H
#include "../agent.h"
#include "../GridWorld/gridworld.h"
#include "../GridWorld/Models/worldmodelgw.h"
#include "../GridWorld/Models/plannergw.h"
TORCH_MODULE(WorldModelGW);

template<class W, class M, class P>
  class ModelBased: public Agent<W>
{
 public:  
  ModelBased();
  ModelBased(W world, M model, P planner);
  void learnWorldModel(string path,int epochs, int batchSize, float lr);
  void saveTrainingData(string dir);
  M getModel() const;
  
 private:
  int modelBatchSize;
  M model;
  P planner;

  vector<float> rLossHistory;
  vector<float> sLossHistory;
};

#endif //MODELBASED_H
