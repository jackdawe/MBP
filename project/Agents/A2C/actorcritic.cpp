#include "actorcritic.h"
#include <chrono>
#include <ctime>
DEFINE_double(g,0.99,"Discount factor");
DEFINE_double(lr,0.003,"Learning Rate");
DEFINE_double(beta,0.01,"Coefficient applied to the entropy loss");
DEFINE_double(zeta,0.5,"Coefficient applied to the value loss");
DEFINE_int32(bs,100,"Batch Size");
DEFINE_int32(n,10000,"Number of training episodes");

template <class W, class M>
ActorCritic<W,M>::ActorCritic()
{
}

template<class W,class M>
ActorCritic<W,M>::ActorCritic(W world, M model,bool usesCNN):
  model(model),usesCNN(usesCNN)
{
  this->controller = world;
  if (usesCNN)
    {
      //     this->generateNameTag("A2C_CNN");
    }
  else
    {
      this->generateNameTag("A2C_MLP");
    }
}


template <class W,class M>
ActorCritic<W,M>::ActorCritic(W world,M model, ParametersA2C param,bool usesCNN):
  Agent<W>(world), model(model), gamma(param.gamma), learningRate(param.learningRate),
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
        torch::Tensor prediction = model->criticOutput(runStates[batchSize-1]
                .reshape({1,3,this->controller.getSize(),this->controller.getSize()}));
        nextReturn = *prediction.to(torch::Device(torch::kCPU)).data<float>();
    }
    runValues[batchSize-1] = nextReturn;
    for (int i=batchSize-2;i>=0;i--)
    {
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
  torch::Tensor actionProbs = model->actorOutput(runStates);
  torch::Tensor valuesEstimate = model->criticOutput(runStates);
  torch::Tensor actionLogProbs = actionProbs.log();
  torch::Tensor chosenActionLogProbs = actionLogProbs.gather(1,runActions.to(torch::kLong)).to(torch::kFloat32);
  torch::Tensor advantages = runValues - valuesEstimate; //TD Error
  torch::Tensor entropy = -(actionProbs*actionLogProbs).sum(1).mean();
  torch::Tensor entropyLoss = beta*entropy;
  torch::Tensor policyLoss = -(chosenActionLogProbs*advantages).mean();
  torch::Tensor valueLoss = zeta*advantages.pow(2).mean();
  torch::Tensor totalLoss = valueLoss + policyLoss - entropyLoss;    
   policyLossHistory.push_back(*policyLoss.to(torch::Device(torch::kCPU)).data<float>());
  valueLossHistory.push_back(*valueLoss.to(torch::Device(torch::kCPU)).data<float>());
  entropyHistory.push_back(*entropyLoss.to(torch::Device(torch::kCPU)).data<float>());
  lossHistory.push_back(*totalLoss.to(torch::Device(torch::kCPU)).data<float>());
  opti->zero_grad();
  totalLoss.backward();
  //vector<torch::Tensor> param = model.parameters();
  //torch::nn::utils::clip_grad_norm_(param,0.5);
  opti->step();
}


template <class W,class M>
void ActorCritic<W,M>::train()
{
  int nSteps = 0;
  this->episodeNumber = 0;
  torch::optim::Adam optimizer(model->parameters(),learningRate);
  while (this->episodeNumber<nEpisodes)
    {
      runStates = torch::zeros(0).to(model->getUsedDevice()), runActions = torch::zeros(0).to(model->getUsedDevice()), runRewards = {}, runAreTerminal = {}, runValues = torch::zeros({batchSize,1}).to(model->getUsedDevice());
      for (int i=0;i<batchSize;i++)
        {
	  nSteps++;
	  torch::Tensor s;
	  if(usesCNN)
            {
	      s = this->controller.toRGBTensor(this->currentState().getStateVector()).to(model->getUsedDevice());	    
            }
	  else
            {
	      s = torch::tensor(this->currentState().getStateVector());
	      s = s.reshape({1,s.size(0)});
            }
	  torch::Tensor actionProbabilities = model->actorOutput(s);
	  torch::Tensor action = actionProbabilities.multinomial(1).to(torch::kFloat32);
	  this->controller.setTakenAction({*action.to(torch::Device(torch::kCPU)).data<float>()});
	  this->controller.setTakenReward(this->controller.transition());
	  runStates = torch::cat({runStates,s});
	  runRewards.push_back(this->takenReward());
	  runActions = torch::cat({runActions,action.to(model->getUsedDevice())});
	  runAreTerminal.push_back(this->controller.isTerminal(this->currentState()));
	  if (runAreTerminal.back() || nSteps==pow(this->controller.getSize(),2)/2)
            {
	      if (nSteps==pow(this->controller.getSize(),2)/2)
		{
		  cout <<"Episode " + to_string(this->episodeNumber) + " was interrupted because it reached the maximum number of steps"<<endl;
		}
	      nSteps = 0;
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
  int count =0;
  while(count<pow(this->controller.getSize(),2) && !this->controller.isTerminal(this->currentState()))
    {
      count++;
      torch::Tensor s;
      if(usesCNN)
            {
	      s = this->controller.toRGBTensor(this->currentState().getStateVector()).to(model->getUsedDevice());	    
            }
	  else
            {
	      s = torch::tensor(this->currentState().getStateVector());
	      s = s.reshape({1,s.size(0)});
            }
	  torch::Tensor actionProbabilities = model->actorOutput(s);
	  torch::Tensor action = actionProbabilities.multinomial(1).to(torch::kFloat32);
	  this->controller.setTakenAction({*action.to(torch::Device(torch::kCPU)).data<float>()});
	  this->controller.setTakenReward(this->controller.transition());
    }
}

template <class W,class M>
void ActorCritic<W,M>::saveTrainingData()
{
  ofstream ag("../PolicyLoss");
  ofstream vl("../ValueLoss");
  ofstream e("../Entropy");
  ofstream tl("../TotalLoss");
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
    }
}

template <class W,class M>
M ActorCritic<W,M>::getModel() const
{
    return model;
}

//template class ActorCritic<GridWorld,ModelA2CGW>;
template class ActorCritic<GridWorld,ConvNetGW>;
