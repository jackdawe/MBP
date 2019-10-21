#include "actorcritic.h"

template <class C, class M>
ActorCritic<C,M>::ActorCritic():
    optimizer(torch::optim::Adam(model.parameters(),learningRate))

{
}

template <class C,class M>
ActorCritic<C,M>::ActorCritic(M model,float gamma, float learningRate, int nEpisodes, int batchSize):
    optimizer(torch::optim::Adam(model.parameters(),learningRate)),gamma(gamma),learningRate(learningRate),
    nEpisodes(nEpisodes),batchSize(batchSize)
{    
}

template <class C,class M>
void ActorCritic<C,M>::updatePolicy()
{
    evaluateRunValues();

    torch::Tensor states = torch::zeros({runStates.size(),runStates[0].size()});

    for (unsigned int i=0;i<runStates.size();i++)
    {
        states[i] = torch::tensor(runStates[i]);
    }
    torch::Tensor actions = torch::zeros({runActions.size(),runActions[0].size()});

    for (unsigned int i=0;i<runActions.size();i++)
    {
        actions[i] = torch::tensor(runActions[i]);
    }


    torch::Tensor actionProbs = model.actorOutput(states);
    torch::Tensor valuesEstimate = model.criticOutput(states);
    torch::Tensor actionLogProbs = actionProbs.log();
    torch::Tensor chosenActionLogProbs = actionLogProbs.gather(1,actions);

    torch::Tensor advantages = torch::tensor(runValues) - valuesEstimate; //TD Error
    torch::Tensor entropy = (actionProbs*actionLogProbs).sum(1).mean();
    torch::Tensor actionGain = (chosenActionLogProbs*advantages).mean();
    torch::Tensor valueLoss = advantages.pow(2).mean();
    torch::Tensor totalLoss = valueLoss - actionGain - 0.0001*entropy;

    optimizer.zero_grad();
    totalLoss.backward();
//    torch::nn::utils::clip_grad_norm_(model.parameters(),0.5);
    optimizer.step();
}


template <class C,class M>
void ActorCritic<C,M>::train()
{
    this->episodeNumber = 0;
    while (this->episodeNumber<nEpisodes)
    {
        runStates = {}, runActions = {}, runRewards = {}, runAreTerminal = {}, runValues = {};

        for (int i=0;i<batchSize;i++)
        {
            torch::Tensor s = torch::tensor(this->currentState().getStateVector());
            torch::Tensor actionProbabilities = model.actorOutput(s);
            torch::Tensor action = actionProbabilities.multinomial(1);
            vector<float> a(action.data<float>(),action.data<float>()+action.numel());
            this->controller.setTakenAction(a);
            this->controller.transition();
            runStates.push_back(this->previousState().getStateVector());
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

template <class C,class M>
void ActorCritic<C,M>::evaluateRunValues()
{
    float nextReturn = 0;
    if (runAreTerminal.back())
    {
        nextReturn = 0;
    }
    else
    {
        //Go through the Critic NN to evaluate the value of the current state
        torch::Tensor s = torch::tensor(this->currentState().getStateVector());
        torch::Tensor co = model.criticOutput(s);
        nextReturn = *co.data<float>();
    }
    runValues.push_back(nextReturn);
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

template class ActorCritic<ControllerGW,ModelA2CGW>;
