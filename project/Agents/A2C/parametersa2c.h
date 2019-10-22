#ifndef PARAMETERSA2C_H
#define PARAMETERSA2C_H
#include <vector>
#include <iostream>
using namespace std;

class ParametersA2C
{
public:
    ParametersA2C();
    ParametersA2C(float gamma, float learningRate, float entropyMultiplier, int batchSize, int nEpisodes,
                  vector<int> mlpHiddenLayers);
    float gamma;
    float learningRate;
    float entropyMultiplier;
    int batchSize;
    int nEpisodes;
    vector<int> mlpHiddenLayers;
    vector<string> names;
};

#endif // PARAMETERSA2C_H
