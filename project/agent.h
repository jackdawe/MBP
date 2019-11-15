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
    Agent(W World, int nEpisodes);
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
    W getWorld() const;

    void setWorld(const W &value);

    string getNameTag() const;

protected:
    int nEpisodes;
    int episodeNumber;
    W world;
    string nameTag;

};

#endif // AGENT_H
