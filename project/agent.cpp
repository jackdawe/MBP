#include "agent.h"

Agent::Agent()
{

}

void Agent::epsilonGreedyPolicy()
{
    default_random_engine generator = default_random_engine(random_device{}());
    uniform_real_distribution<float> dist(0,1);
    if (dist(generator) < epsilon)
    {
        for (int i=0;i<actionSpace.size();i++)
        {
            takenAction[i] = actionSpace[i].pick();
        }
    }
}

void Agent::initialiseEpisode()
{
}

void Agent::policy()
{
}

void Agent::updatePolicy()
{
}

void Agent::savePolicy()
{
}

void Agent:: loadPolicy()
{
}

void Agent::saveTrainingData()
{
}

void Agent::addToRewardHistory(double r)
{
    rewardHistory.push_back(r);
}
