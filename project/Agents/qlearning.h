#ifndef QLEARNING_H
#define QLEARNING_H
#include "agent.h"

template <class C>
class QLearning: public Agent<C>
{
public:
    QLearning();
    QLearning(C controller,int nEpisodes, float epsilon, float gamma);
    void epsilonGreedyPolicy();
    void updateQValues();
    void train();
    void playOne();
    void saveQValues();
    void loadQValues(string tag);
    void saveTrainingData();

private:
    int nEpisodes;
    float epsilon;
    float gamma;
    vector<vector<float>> qvalues;
};

#endif // QLEARNING_H
