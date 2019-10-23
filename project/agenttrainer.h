#ifndef AGENTTRAINER_H
#define AGENTTRAINER_H
#include "Agents/qlearning.h"
#include "GridWorld/episodeplayergw.h"
#include <iostream>

template <class A>
class AgentTrainer
{
public:
    AgentTrainer();
    void train(A* agent, int numberOfEpisodes,int trainMode = 0,int savingSequenceMode = 0);
    void displayScoresGW(string mapTag, string policyTag);
};

#endif // AGENTTRAINER_H
