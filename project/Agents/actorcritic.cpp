#include "actorcritic.h"

template <class C>
ActorCritic<C>::ActorCritic()
{

}

template <class C>
void ActorCritic<C>::updatePolicy()
{
    vector<float>
}


template <class C>
void ActorCritic<C>::train()
{
    episodeNumber = 0;
    while (episodeNumber<nEpisodes)
    {
        runStates = {}, runActions = {}, runRewards = {}, runAreTerminal = {}, runValues = {};

        for (int i=0;i<batchSize;i++)
        {
            epsilonGreedyPolicy();
            runStates.push_back(previousState());
            runRewards.push_back(takenReward());
            runActions.push_back(takenAction());
            runAreTerminal.push_back(controller.isTerminal(currentState()));

            if (runAreTerminal.back())
            {
                controller.reset();
                episodeNumber++;
            }
        }
        updatePolicy();
    }
}

template <class C>
void ActorCritic<C>::evaluateRunValues()
{
    float nextReturn = 0;
    if (runAreTerminal.back())
    {
        nextReturn = 0;
    }
    else
    {
        //Go through the Critic NN to evaluate the value of the last state
        //nextReturn = criticNN();
        runValues.append(nextReturn);
    }
    //GO backwards to evaluate the value of previous states using the estimated values of the future states
    for (int i=batchSize-1;i>=0;i--)
    {
        if (runAreTerminal[i])
        {
            nextReturn=0;
        }
        else
        {
            nextReturn=runRewards[i] + gamma*nextReturn;
        }
        runValues.push_back(nextReturn);
    }
}
