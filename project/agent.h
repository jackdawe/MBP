#ifndef AGENT_H
#define AGENT_H
#include "state.h"
#include <vector>
#include <fstream>
#include <iostream>

template <class T>
class Agent
{
public:
    Agent();
    Agent(vector<Action> actionSpace, float epsilon);
    virtual void initialiseEpisode();
    void epsilonGreedyPolicy();
    virtual void policy();
    virtual void updatePolicy();
    virtual void savePolicy();
    virtual void loadPolicy();
    virtual void saveTrainingData();
    void addToRewardHistory(double r);
    void generateNameTag(vector<float> parameters, vector<string> parametersName);

private:
    int episodeNumber;
    float epsilon;
    T previousState;
    vector<Action> actionSpace;
    vector<double> takenAction;
    double takenReward;
    T nextState;
    vector<double> rewardHistory;
    string nameTag;

};

#endif // AGENT_H
