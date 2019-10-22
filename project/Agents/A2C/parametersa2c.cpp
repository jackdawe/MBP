#include "parametersa2c.h"

ParametersA2C::ParametersA2C(){}

ParametersA2C::ParametersA2C(float gamma, float learningRate, float entropyMultiplier, int batchSize, int nEpisodes, vector<int> mlpLayers):
    gamma(gamma), learningRate(learningRate),entropyMultiplier(entropyMultiplier), batchSize(batchSize),
    nEpisodes(nEpisodes),mlpHiddenLayers(mlpLayers),
    names({"GAMMA","LEARNING RATE", "ENTROPY MULTIPLIER", "BATCH SIZE","NUMBER OF TRAINING EPISODES","NUMBER OF HIDDEN UNITS IN EACH FC LAYERS"})
{}