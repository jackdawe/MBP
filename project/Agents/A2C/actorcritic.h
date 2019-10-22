#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H
#include "agent.h"
#include "parametersa2c.h"
#include "../../GridWorld/modela2cgw.h"

template <class C, class M>
class ActorCritic: public Agent<C>
{
public:
    ActorCritic();
    ActorCritic(C Controller, ParametersA2C param);
    void updatePolicy();
    void train();
    void evaluateRunValues();
    M getModel() const;

private:
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
};

#endif // ACTORCRITIC_H
