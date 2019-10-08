#ifndef QLEARNINGAGENT_H
#define QLEARNINGAGENT_H
#include "agent.h"

template <class T>
class QLearningAgent: public Agent<T>
{
public:
    QLearningAgent();
    QLearningAgent(vector<Action> actionSpace, float epsilon, float gamma);
    void greedyPolicy();
    void updatePolicy();
    void savePolicy();
    void loadPolicy();
    void saveTrainingData();

private:
    float gamma;
    vector<double> qvalues;
};

#endif // QLEARNINGAGENT_H
