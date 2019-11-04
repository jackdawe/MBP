#include "actorcritic.h"

template <class W, class M>
ActorCritic<W,M>::ActorCritic()
{
}

template <class W,class M>
ActorCritic<W,M>::ActorCritic(W controller,M model, ParametersA2C param,bool usesCNN):
    Agent<W>(controller), model(model), gamma(param.gamma), learningRate(param.learningRate),
    beta(param.beta),zeta(param.zeta), nEpisodes(param.nEpisodes),batchSize(param.batchSize), usesCNN(usesCNN)
{
    if (usesCNN)
    {
        this->generateNameTag("A2C_CNN");
    }
    else
    {
        this->generateNameTag("A2C_MLP");
    }
}

template <class W,class M>
void ActorCritic<W,M>::evaluateRunValues()
{
    float nextReturn;
    float thisReturn;
    if (runAreTerminal[batchSize-1])
    {
        nextReturn = runRewards[batchSize-1];
    }
    else
    {
        torch::Tensor prediction = model.criticOutput(runStates[batchSize-1]
                .reshape({1,3,this->controller.getSize(),this->controller.getSize()}));
        nextReturn = *prediction.data<float>();
    }
    runValues[batchSize-1] = nextReturn;
    for (int i=batchSize-2;i>=0;i--)
    {
//        if (*runStates[i][1][3][3].data<float>() == 1)
//        {
//            cout << i << endl;
//        }
//        cout << runStates[batchSize-2-i][0] << endl;
        if (runAreTerminal[i])
        {
            nextReturn=0;
        }
        thisReturn=runRewards[i] + gamma*nextReturn;
        runValues[i][0] = thisReturn;
        nextReturn = thisReturn;        
    }
}

template <class W,class M>
void ActorCritic<W,M>::backPropagate(torch::optim::Adam *opti)
{
    evaluateRunValues();    
    torch::Tensor actionProbs = model.actorOutput(runStates);
    torch::Tensor valuesEstimate = model.criticOutput(runStates);
    torch::Tensor actionLogProbs = actionProbs.log();
    torch::Tensor chosenActionLogProbs = actionLogProbs.gather(1,runActions.to(torch::kLong)).to(torch::kFloat32);
    torch::Tensor advantages = runValues - valuesEstimate; //TD Error

    GridWorld gw("../GridWorld/MapPools/8x8/Easy/Train/map3",1,0);
    gw.generateVectorStates();
    torch::Tensor value = model.criticOutput(gw.toRGBTensor(gw.getCurrentState().getStateVector()));

    torch::Tensor entropy = -(actionProbs*actionLogProbs).sum(1).mean();
    torch::Tensor entropyLoss = beta*entropy;
    torch::Tensor policyLoss = -(chosenActionLogProbs*advantages).mean();
    torch::Tensor valueLoss = zeta*advantages.pow(2).mean();
    torch::Tensor totalLoss = valueLoss + policyLoss - entropyLoss;

    vHistory.push_back(*value.data<float>());
    policyLossHistory.push_back(*policyLoss.data<float>());
    valueLossHistory.push_back(*valueLoss.data<float>());
    entropyHistory.push_back(*entropyLoss.data<float>());
    lossHistory.push_back(*totalLoss.data<float>());

    opti->zero_grad();
    totalLoss.backward();
    vector<torch::Tensor> param = model.parameters();
    torch::nn::utils::clip_grad_norm_(param,0.5);
    opti->step();
}


template <class W,class M>
void ActorCritic<W,M>::train()
{
    this->episodeNumber = 0;
    torch::optim::Adam optimizer(model.parameters(),learningRate);
    while (this->episodeNumber<nEpisodes)
    {
        runStates = torch::zeros(0), runActions = torch::zeros(0), runRewards = {}, runAreTerminal = {}, runValues = torch::zeros({batchSize,1});
        for (int i=0;i<batchSize;i++)
        {
            torch::Tensor s;
            if(usesCNN)
            {
                s = this->controller.toRGBTensor(this->currentState().getStateVector());
            }
            else
            {
                s = torch::tensor(this->currentState().getStateVector());
                s = s.reshape({1,s.size(0)});
            }
            torch::Tensor actionProbabilities = model.actorOutput(s);
            torch::Tensor action = actionProbabilities.multinomial(1).to(torch::kFloat32);            
            this->controller.setTakenAction({*action.data<float>()});
            this->controller.setTakenReward(this->controller.transition());
            runStates = torch::cat({runStates,s});
            runRewards.push_back(this->takenReward());
            runActions = torch::cat({runActions,action});
            runAreTerminal.push_back(this->controller.isTerminal(this->currentState()));

            if (runAreTerminal.back())
            {

                this->controller.reset();
                this->episodeNumber++;
                //Displaying a progression bar in the terminal

                if (nEpisodes > 100 && lossHistory.size()>0 &&  this->episodeNumber%(nEpisodes/100) == 0)
                {
                    cout << "Training in progress... " + to_string(this->episodeNumber/(nEpisodes/100)) + "%. Current Loss: " + to_string(lossHistory.back())
                         + "  Current entropy: " + to_string(entropyHistory.back()/beta)<< endl;
                }
            }
        }

        backPropagate(&optimizer);
    }
    saveTrainingData();
    this->controller.saveRewardHistory("A2C");
}

template <class W, class M>
void ActorCritic<W,M>::playOne()
{
    while(!this->controller.isTerminal(this->currentState()))
    {
        torch::Tensor s;
        if(usesCNN)
        {
            s = this->controller.toRGBTensor(this->previousState().getStateVector());
        }
        else
        {
            s = torch::tensor(this->previousState().getStateVector());
        }
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
    ofstream ag("../PolicyLoss");
    ofstream vl("../ValueLoss");
    ofstream e("../Entropy");
    ofstream tl("../TotalLoss");
    ofstream v("../Values");
    if(!ag)
    {
        cout<<"oups"<<endl;
    }
    for (unsigned int i=0;i<policyLossHistory.size();i++)
    {
        ag<<to_string(policyLossHistory[i])<<endl;
        vl<<to_string(valueLossHistory[i]) <<endl;
        e<<to_string(entropyHistory[i])<<endl;
        tl<<to_string(lossHistory[i])<<endl;
        v<<to_string(vHistory[i])<<endl;
    }
}

template <class W,class M>
M ActorCritic<W,M>::getModel() const
{
    return model;
}

template class ActorCritic<GridWorld,ModelA2CGW>;
template class ActorCritic<GridWorld,ConvNetGW>;
