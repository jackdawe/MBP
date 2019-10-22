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
    Agent(C controller, float epsilon);
    virtual void initialiseEpisode();
    void epsilonGreedyPolicy();
    virtual void greedyPolicy();
    virtual void updatePolicy();
    virtual void finaliseEpisode();
    virtual void savePolicy();
    virtual void loadPolicy(string tag);
    void generateNameTag(string prefix);
    void resetController();
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
    float epsilon;

    string getNameTag() const;

protected:
    int episodeNumber;
    C controller;
    string nameTag;

};

#endif // AGENT_H
