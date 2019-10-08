#ifndef QLEARNING_H
#define QLEARNING_H
#include "agent.h"

template <class T>
class QLearning: public Agent<T>
{
public:
    QLearning();
    QLearning(vector<Action> actionSpace, float epsilon, float gamma);
    void greedyPolicy();
    void updatePolicy();
    void savePolicy();
    void loadPolicy();
    void saveTrainingData();

private:
    float gamma;
    vector<double> qvalues;
};

#endif // QLEARNING_H
