#include "parametersa2c.h"

ParametersA2C::ParametersA2C(){}

ParametersA2C::ParametersA2C(float gamma, float learningRate, float entropyMultiplier, int batchSize, int nEpisodes):
    gamma(gamma), learningRate(learningRate),entropyMultiplier(entropyMultiplier), batchSize(batchSize),
    nEpisodes(nEpisodes),
    names({"GAMMA","LEARNING RATE", "ENTROPY MULTIPLIER", "BATCH SIZE","NUMBER OF TRAINING EPISODES"})
{}
