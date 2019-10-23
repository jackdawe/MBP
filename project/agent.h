#ifndef AGENT_H
#define AGENT_H
#include "GridWorld/controllergw.h"
#include "Starship/controllerss.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

template <class C>
class Agent
{
public:
    Agent();
    Agent(C Controller);
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
    C getController() const;

    void setController(const C &value);

    string getNameTag() const;

protected:
    int episodeNumber;
    C controller;
    string nameTag;

};

#endif // AGENT_H
