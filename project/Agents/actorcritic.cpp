#include "actorcritic.h"
#include <torch/torch.h>
template <class C>
ActorCritic<C>::ActorCritic()
{

}

template <class C>
void ActorCritic<C>::updatePolicy()
{
    evaluateRunValues();
    //Do things using pytorch
}


template <class C>
void ActorCritic<C>::train()
{
    this->episodeNumber = 0;
    while (this->episodeNumber<nEpisodes)
    {
        runStates = {}, runActions = {}, runRewards = {}, runAreTerminal = {}, runValues = {};

        for (int i=0;i<batchSize;i++)
        {
            this->epsilonGreedyPolicy();
            runStates.push_back(this->previousState());
            runRewards.push_back(this->takenReward());
            runActions.push_back(this->takenAction());
            runAreTerminal.push_back(this->controller.isTerminal(this->currentState()));

            if (runAreTerminal.back())
            {
                this->controller.reset();
                this->episodeNumber++;
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
        runValues.push_back(nextReturn);
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
