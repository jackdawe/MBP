#include "actorcritic.h"

template <class C, class M>
ActorCritic<C,M>::ActorCritic():
    optimizer(torch::optim::Adam(model.parameters(),learningRate))

{
}

template <class C,class M>
ActorCritic<C,M>::ActorCritic(C controller, ParametersA2C param):
    Agent<C>(controller),model(ModelA2CGW(this->controller.getCurrentState().getStateVector().size(),param.mlpHiddenLayers[0],param.mlpHiddenLayers[1],param.mlpHiddenLayers[2])),
    optimizer(torch::optim::Adam(model.parameters(),param.learningRate)),gamma(param.gamma),
    learningRate(param.learningRate),entropyMultiplier(param.entropyMultiplier), nEpisodes(param.nEpisodes),batchSize(param.batchSize)
{
    this->generateNameTag("A2C_MLP");
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
    torch::Tensor chosenActionLogProbs = actionLogProbs.gather(1,actions.to(torch::kLong)).to(torch::kFloat32);

    torch::Tensor advantages = torch::tensor(runValues) - valuesEstimate; //TD Error
    torch::Tensor entropy = (actionProbs*actionLogProbs).sum(1).mean();
    torch::Tensor actionGain = (chosenActionLogProbs*advantages).mean();
    torch::Tensor valueLoss = advantages.pow(2).mean();
    torch::Tensor totalLoss = valueLoss - actionGain - entropyMultiplier*entropy;

    //Displaying a progression bar in the terminal

    if (true)//nEpisodes > 100 && this->episodeNumber%(5*nEpisodes/100) == 0)
    {
        cout << "Training in progress... " + to_string(this->episodeNumber/(nEpisodes/100)) + "%. Current Loss: " + to_string(*totalLoss.data<float>())
             + "  Current entropy: " + to_string(*entropy.data<float>())<< endl;
    }
    optimizer.zero_grad();
    totalLoss.backward();
    vector<torch::Tensor> param = model.parameters();
    torch::nn::utils::clip_grad_norm_(param,0.5);
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
            torch::Tensor s = torch::tensor(this->previousState().getStateVector());
            torch::Tensor actionProbabilities = model.actorOutput(s.reshape({1,s.size(0)}));
            torch::Tensor action = actionProbabilities.multinomial(1).to(torch::kFloat32);
            vector<float> a(action.data<float>(),action.data<float>()+action.numel());

            this->controller.setTakenAction(a);
            this->controller.setTakenReward(this->controller.transition());
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
        nextReturn = runRewards.back();
    }
    else
    {
        //Go through the Critic layer to evaluate the value of the current state
        torch::Tensor s = torch::tensor(this->previousState().getStateVector());
        torch::Tensor co = model.criticOutput(s.reshape({1,s.size(0)}));
        nextReturn = *co.data<float>();
    }
    runValues.push_back(nextReturn);
    //GO backwards to evaluate the value of previous states using the estimated values of the future states
    float thisReturn = 0;
    for (int i=batchSize-2;i>=0;i--)
    {
        if (runAreTerminal[i])
        {
            nextReturn=0;
        }
        thisReturn=runRewards[i] + gamma*nextReturn;

        runValues.push_back(thisReturn);
        nextReturn = thisReturn;
    }
    reverse(runValues.begin(),runValues.end());
}

template <class C,class M>
M ActorCritic<C,M>::getModel() const
{
    return model;
}

template class ActorCritic<ControllerGW,ModelA2CGW>;
