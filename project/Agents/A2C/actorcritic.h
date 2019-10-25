#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H
#include "agent.h"
#include "parametersa2c.h"
#include "../../GridWorld/Models/modela2cgw.h"
#include "../../GridWorld/Models/convnetgw.h"

template <class W, class M>
class ActorCritic: public Agent<W>
{
public:
    ActorCritic();
    ActorCritic(W World, M model, ParametersA2C param, bool usesCNN = false);
    void evaluateRunValues();
    void backPropagate(torch::optim::Adam *opti);
    void train();
    void playOne();    
    void saveTrainingData();
    M getModel() const;

private:
    bool usesCNN;
    float gamma;
    float learningRate;
    float beta;
    float zeta;
    int nEpisodes;
    int batchSize;    
    torch::Tensor runStates;
    torch::Tensor runActions;
    vector<float> runRewards;
    vector<bool> runAreTerminal;
    torch::Tensor runValues;
    M model;

    vector<float> actionGainHistory;
    vector<float> valueLossHistory;
    vector<float> entropyHistory;
    vector<float> lossHistory;
};

#endif // ACTORCRITIC_H
