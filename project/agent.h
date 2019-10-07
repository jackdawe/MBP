#ifndef AGENT_H
#define AGENT_H
#include "state.h"
#include "action.h"
#include <vector>

using namespace std;

class Agent
{
public:
    Agent();
    virtual void initialiseEpisode();
    void epsilonGreedyPolicy();
    virtual void policy();
    virtual void updatePolicy();
    virtual void savePolicy();
    virtual void loadPolicy();
    virtual void saveTrainingData();

private:
    int episodeNumber;
    float epsilon;
    State initialState;
    State previousState;
    vector<Action> actionSpace;
    vector<double> takenAction;
    double takenReward;
    State nextState;
    vector<double> rewardHistory;
    string nameTag;

};

#endif // AGENT_H
