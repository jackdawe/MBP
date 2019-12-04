#ifndef QLEARNING_H
#define QLEARNING_H
#include "agent.h"

template <class W>
class QLearning: public Agent<W>
{
public:
    QLearning();
    QLearning(W world);
    void epsilonGreedyPolicy(float e);
    void updateQValues();
    void train(int nEpisodes, float epsilon, float gamma);
    void playOne();
    void saveQValues(string filename);
    void loadQValues(string filename);
    void saveTrainingData();

private:
    int episodeId;
    float epsilon;
    float gamma;
    vector<vector<float>> qvalues;
};

#endif // QLEARNING_H
