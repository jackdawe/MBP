#include "parametersa2c.h"

ParametersA2C::ParametersA2C(){}

ParametersA2C::ParametersA2C(float gamma, float learningRate, float beta,float zeta, int batchSize, int nEpisodes):
    gamma(gamma), learningRate(learningRate),beta(beta),zeta(zeta), batchSize(batchSize),
    nEpisodes(nEpisodes),
    names({"GAMMA","LEARNING RATE", "BETA","ZETA", "BATCH SIZE","NUMBER OF TRAINING EPISODES"})
{}
