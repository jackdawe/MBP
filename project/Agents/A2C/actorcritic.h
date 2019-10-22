#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H
#include "agent.h"
#include "../../GridWorld/modela2cgw.h"

template <class C, class M>
class ActorCritic: public Agent<C>
{
public:
    ActorCritic();
    ActorCritic(C Controller, M model, float gamma, float learningRate, int nEpisodes, int batchSize);
    void updatePolicy();
    void train();
    void evaluateRunValues();
    M getModel() const;

private:
    float gamma;
    float learningRate;
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
