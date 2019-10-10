#ifndef AGENT_H
#define AGENT_H
#include "GridWorld/controllergw.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

template <class C>
class Agent
{
public:
    Agent();
    Agent(C controller, float epsilon);
    virtual void initialiseEpisode();
    void epsilonGreedyPolicy();
    virtual void greedyPolicy();
    virtual void updatePolicy();
    virtual void savePolicy(string path);
    virtual void loadPolicy(string filename);
    void generateNameTag(vector<float> parameters, vector<string> parametersName);

    int daSize();
    int caSize();
    ActionSpace actions();
    vector<DiscreteAction> discreteActions();
    vector<ContinuousAction> continuousActions();
    State previousState();
    vector<double> takenAction();
    double takenReward();
    State currentState();
    vector<double> rewardHistory();
    C getController() const;
    void addToRewardHistory(double r);

protected:
    int episodeNumber;
    float epsilon;
    C controller;
    string nameTag;

};

#endif // AGENT_H
