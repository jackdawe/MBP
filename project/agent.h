#ifndef AGENT_H
#define AGENT_H
#include "GridWorld/gridworld.h"
#include "Starship/spaceworld.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

template <class W>
class Agent
{
public:
    Agent();
    Agent(W World);
    void saveRewardHistory();
    void saveLastEpisode();
    void loadEpisode(string nameTag);

    int daSize();
    int caSize();
    ActionSpace actions();
    vector<DiscreteAction> discreteActions();
    vector<ContinuousAction> continuousActions();
    State previousState();
    vector<float> takenAction();
    float takenReward();
    State currentState();
    vector<float> rewardHistory();
    void resetWorld();
    W getWorld() const;    
    void setWorld(const W &value);

protected:
    W world;
};

#endif // AGENT_H
