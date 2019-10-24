#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H
#include "agent.h"
#include "parametersa2c.h"
#include "../../GridWorld/modela2cgw.h"
#include "../../GridWorld/convnetgw.h"

template <class W, class M>
class ActorCritic: public Agent<W>
{
public:
    ActorCritic();
    ActorCritic(W World, ParametersA2C param, bool usesCNN = false);
    void evaluateRunValues();
    void backPropagate();
    void train();
    void playOne();    
    void saveTrainingData();
    M getModel() const;

private:
    bool usesCNN;
    float gamma;
    float learningRate;
    float entropyMultiplier;
    int nEpisodes;
    int batchSize;    
    vector<vector<float>> runStates;
    vector<vector<float>> runActions;
    vector<float> runRewards;
    vector<bool> runAreTerminal;
    vector<float> runValues;
    M model;
    torch::optim::Adam optimizer;

    vector<float> actionGainHistory;
    vector<float> valueLossHistory;
    vector<float> entropyHistory;
    vector<float> lossHistory;
};

#endif // ACTORCRITIC_H
