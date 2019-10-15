#ifndef AGENTTRAINER_H
#define AGENTTRAINER_H
#include "Agents/qlearning.h"
#include "Agents/randomagent.h"
#include <iostream>

template <class A>
class AgentTrainer
{
public:
    AgentTrainer();
    void train(A* agent, int numberOfEpisodes,int trainMode = 0,int savingSequenceMode = 0);
    void saveEpisode(A agent, int seqId);
    void loadSequence(int seqId);
    vector<float> loadParameters(int seqId);
    vector<vector<float> > getStateSequence() const;
    vector<vector<float> > getActionSequence() const;
    vector<float> getParameters() const;

private:
    vector<vector<float>> stateSequence;
    vector<vector<float>> actionSequence;
    vector<float> parameters;
};

#endif // AGENTTRAINER_H
