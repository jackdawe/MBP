#ifndef QLEARNING_H
#define QLEARNING_H
#include "agent.h"

template <class C>
class QLearning: public Agent<C>
{
public:
    QLearning();
    QLearning(C controller, float epsilon, float gamma);
    void greedyPolicy();
    void updatePolicy();
    void savePolicy(string path);
    void loadPolicy(string filename);
    void saveTrainingData();

private:
    float gamma;
    vector<double> qvalues;
};

#endif // QLEARNING_H
