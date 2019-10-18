#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H
#include "agent.h"

template <class C>
class ActorCritic: public Agent<C>
{
public:
    ActorCritic();
    void updatePolicy();
    void train();
    void evaluateRunValues();
private:
    float gamma;
    float learningRate;
    float nEpisodes;
    float batchSize;
    vector<vector<float>> runStates;
    vector<vector<float>> runActions;
    vector<float> runRewards;
    vector<bool> runAreTerminal;
    vector<float> runValues;
    //Neural Network model
    //Optimizer
};

#endif // ACTORCRITIC_H
