#include "actorcritic.h"

template <class W, class M>
ActorCritic<W,M>::ActorCritic():
    optimizer(torch::optim::Adam(model.parameters(),learningRate))

{
}

template <class W,class M>
ActorCritic<W,M>::ActorCritic(W controller, ParametersA2C param):
    Agent<W>(controller),model(ModelA2CGW(this->controller.getCurrentState().getStateVector().size(),param.mlpHiddenLayers[0],param.mlpHiddenLayers[1],param.mlpHiddenLayers[2])),
    optimizer(torch::optim::Adam(model.parameters(),param.learningRate)),gamma(param.gamma),
    learningRate(param.learningRate),entropyMultiplier(param.entropyMultiplier), nEpisodes(param.nEpisodes),batchSize(param.batchSize)
{
    this->generateNameTag("A2C_MLP");
}

template <class W,class M>
void ActorCritic<W,M>::evaluateRunValues()
{
    float nextReturn = 0;
    float thisReturn = 0;
    for (int i=batchSize-1;i>=0;i--)
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

template <class W,class M>
void ActorCritic<W,M>::backPropagate()
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
    torch::Tensor entropy = -(actionProbs*actionLogProbs).sum(1).mean();
    torch::Tensor actionGain = (chosenActionLogProbs*advantages).mean();
    torch::Tensor valueLoss = advantages.pow(2).mean();
    torch::Tensor totalLoss = valueLoss - actionGain + entropyMultiplier*entropy;

    actionGainHistory.push_back(*actionGain.data<float>());
    valueLossHistory.push_back(*valueLoss.data<float>());
    entropyHistory.push_back(*entropy.data<float>());
    lossHistory.push_back(*totalLoss.data<float>());

    optimizer.zero_grad();
    totalLoss.backward();
    vector<torch::Tensor> param = model.parameters();
    torch::nn::utils::clip_grad_norm_(param,0.5);
    optimizer.step();
}


template <class W,class M>
void ActorCritic<W,M>::train()
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
                //Displaying a progression bar in the terminal

                if (nEpisodes > 100 && this->episodeNumber%(5*nEpisodes/100) == 0)
                {
                    cout << "Training in progress... " + to_string(this->episodeNumber/(nEpisodes/100)) + "%. Current Loss: " + to_string(lossHistory.back())
                         + "  Current entropy: " + to_string(entropyHistory.back())<< endl;
                }
            }
        }
        backPropagate();
    }
    saveTrainingData();
    this->controller.saveRewardHistory("A2C");
}

template <class W, class M>
void ActorCritic<W,M>::playOne()
{
    while(!this->controller.isTerminal(this->currentState()))
    {
        torch::Tensor s = torch::tensor(this->previousState().getStateVector());
        torch::Tensor actionProbabilities = model.actorOutput(s.reshape({1,s.size(0)}));
        torch::Tensor action = actionProbabilities.multinomial(1).to(torch::kFloat32);
        vector<float> a(action.data<float>(),action.data<float>()+action.numel());
        this->controller.setTakenAction(a);
        this->controller.setTakenReward(this->controller.transition());
    }
    this->saveLastEpisode();
}

template <class W,class M>
void ActorCritic<W,M>::saveTrainingData()
{
    ofstream ag("../ActionGain");
    ofstream vl("../ValueLoss");
    ofstream e("../Entropy");
    ofstream tl("../TotalLoss");
    if(!ag)
    {
        cout<<"oups"<<endl;
    }
    for (unsigned int i=0;i<actionGainHistory.size();i++)
    {
        ag<<to_string(actionGainHistory[i])<<endl;
        vl<<to_string(valueLossHistory[i]) <<endl;
        e<<to_string(entropyHistory[i])<<endl;
        tl<<to_string(lossHistory[i])<<endl;
    }
}

template <class W,class M>
M ActorCritic<W,M>::getModel() const
{
    return model;
}

template class ActorCritic<GridWorld,ModelA2CGW>;
