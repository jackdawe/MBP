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
    void generateNameTag(string prefix);
    void incrementEpisode();
    void saveRewardHistory();
    void saveLastEpisode();
    void loadEspisode(string nameTag);

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
    W getController() const;

    void setController(const W &value);

    string getNameTag() const;

protected:
    int episodeNumber;
    W controller;
    string nameTag;

};

#endif // AGENT_H
