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
    void finaliseEpisode();
    void savePolicy();
    void loadPolicy(string tag);
    void saveTrainingData();

private:
    float gamma;
    vector<float> qvalues;
};

#endif // QLEARNING_H
