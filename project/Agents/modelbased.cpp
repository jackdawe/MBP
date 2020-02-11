#include "modelbased.h"
template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased():
  device(torch::Device(torch::kCPU))
{
}

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased(W world, F forwardModel):
  forwardModel(forwardModel), device(forwardModel->usedDevice), epoch(0)
{
  this->world = world;
}

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased(W world, F forwardModel, P planner):
  forwardModel(forwardModel),planner(planner),device(forwardModel->usedDevice), epoch(0)
{
  this->world = world;
}

template <class W, class F, class P>
void ModelBased<W,F,P>::learnForwardModel(torch::optim::Adam *optimizer, torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize, float beta)
{
  int n = stateInputs.size(0);
  
  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      epoch++;
      for (int i=0;i<n/batchSize;i++)
	{
	  //Extracting batch from dataset
	  
	  int nTimesteps = rewardLabels.size(1);
	  torch::Tensor siBatch = torch::zeros(0);
	  torch::Tensor aiBatch = torch::zeros(0);
	  torch::Tensor slBatch = torch::zeros(0);
	  torch::Tensor rlBatch = torch::zeros({batchSize,nTimesteps});
	  for (int i=0;i<batchSize;i++)
	    {
	      int index = dist(generator);
	      siBatch = torch::cat({siBatch,stateInputs[index].unsqueeze(0)});
	      aiBatch = torch::cat({aiBatch,actionInputs[index].unsqueeze(0)});
	      slBatch = torch::cat({slBatch,stateLabels[index].unsqueeze(0)});
	      rlBatch[i] = rewardLabels[index]; 
	    }
	  //Forward and backward pass
	  
	  forwardModel->forward(siBatch,aiBatch);
	  forwardModel->computeLoss(slBatch,rlBatch);
	  torch::Tensor sLoss = beta*forwardModel->stateLoss, rLoss = forwardModel->rewardLoss;
	  torch::Tensor totalLoss = sLoss+rLoss;
	  optimizer->zero_grad();
	  totalLoss.backward();
	  optimizer->step();
	  sLossHistory.push_back(*sLoss.to(torch::Device(torch::kCPU)).data<float>());
	  rLossHistory.push_back(*rLoss.to(torch::Device(torch::kCPU)).data<float>());
      
	  //Printing some stuff
	  
	  if (i%10 == 0 || i == n/batchSize-1)
	    {	  
	      cout<< "Epoch " + to_string(epoch) + " - Iteration " + to_string(i+1)+"/"+to_string(n/batchSize)+" : " + to_string(*totalLoss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	    }
	}
    }
}

template <class W, class F, class P>
void ModelBased<W,F,P>::gradientBasedPlanner(torch::Tensor initState, ActionSpace actionSpace, int nRollouts, int nTimesteps, int nGradsteps, float lr, torch::Tensor initActions)
{
  ofstream f1("../temp/rewIllu");
  ofstream f2("../temp/rewTrue");
  
  //Setting everything up
  
  unsigned int s = initState.size(0);
  torch::Tensor stateSequences = torch::zeros(0);  
  torch::Tensor costs = torch::zeros({nRollouts});
  
  unsigned int nca = actionSpace.getContinuousActions().size();
  unsigned int nda = actionSpace.nActions()-nca;
  vector<DiscreteAction> discreteActions = actionSpace.getDiscreteActions();

  //Randomly initialising discrete and continuous actions and merging them

  torch::Tensor iactions = initActions;
  if (torch::equal(initActions,torch::zeros(0)))
    {
      torch::Tensor initTokens = getInitTokens(discreteActions, nTimesteps, nRollouts);
      torch::Tensor initCA = getInitCA(nca, nTimesteps, nRollouts);
      iactions = torch::cat({initTokens, initCA},2);
    }
  torch::autograd::Variable actions(iactions.clone().set_requires_grad(true));
  vector<torch::Tensor> aa;
  aa.push_back(actions);
  torch::optim::Adam optimizer(aa,lr);
  
  torch::Tensor savedCA = actions.slice(2,nda,nda+nca,1); //TO REMOVE
  
  //Looping over the number of optimisation steps 

  for (int i=0;i<nGradsteps+1;i++)
    {            
      stateSequences = initState.expand({nRollouts,s});
      
      //Putting each discrete action into a vector and putting continuous actions into one tensor  
      
      torch::Tensor toOptiDA = actions.slice(2,0,nda,1);
      torch::Tensor toOptiCA = torch::clamp(actions.slice(2,nda,nda+nca,1),0,1);
      vector<torch::Tensor> daTokens = breakDAIntoTokens(discreteActions,toOptiDA);      

      //Predicting the state sequence given the initState and the action sequence
      {
	torch::NoGradGuard no_grad;

	//Clipping tokens to closest one-hot encoded vector to minimise inference error
	
	torch::Tensor ohev = tokensToOneHot(daTokens,discreteActions.size());
	ohev = torch::cat({ohev,toOptiCA},-1);	    	
	forwardModel->forward(stateSequences,ohev.transpose(0,1));
	stateSequences = torch::cat({stateSequences.unsqueeze(1).to(device), forwardModel->predictedStates},1);
      }
      
      //Predicting the rewards given the state and action sequences 

      f2<<computeTrueReward(initState, discreteActions, actions, actionSpace.getContinuousActions(), nca,nda,nTimesteps)<<endl; //TO REMOVE  

      if (i<nGradsteps)
	{
	  torch::Tensor softmaxActions = torch::zeros(0);
	  for (int d=0;d<discreteActions.size();d++)
	    {
	      softmaxActions = torch::cat({softmaxActions,torch::softmax(daTokens[d],2)},2);
	    }
	  softmaxActions = torch::cat({softmaxActions,toOptiCA},2);       
	  torch::Tensor reward = torch::zeros({nTimesteps,nRollouts}).to(device); 
	  for (int t=0;t<nTimesteps;t++)
	    {
	      forwardModel->forward(stateSequences.transpose(0,1)[t],softmaxActions[t]);
	      reward[t] = forwardModel->predictedRewards.squeeze();
	    }
	  optimizer.zero_grad();
	  costs = -reward.sum(0);	
	  cout<<-costs.mean()<<endl;
	  f1<<-*costs.to(torch::Device(torch::kCPU)).mean().data<float>()<<endl; //TO REMOVE
	  costs.backward(torch::ones({nRollouts}).to(device));
	  optimizer.step();
	}
    }
  actions = actions.transpose(0,1);
  
  //Decoding action tensors back to their original form

  torch::Tensor finalDA = getFinalDA(discreteActions, actions.slice(2,0,nda,1));
  torch::Tensor finalCA = getFinalCA(actionSpace.getContinuousActions(), torch::clamp(actions.slice(2,nda,nda+nca,1),0,1));
  torch::Tensor actionSequences = torch::cat({finalDA,finalCA},2);
  int maxRewardIdx = *torch::argmax(-costs.to(torch::Device(torch::kCPU))).data<long>();
  cout<<"......"<<endl;
  //cout<<actionSequences[maxRewardIdx]<<endl;      
  cout<<-costs[maxRewardIdx]<<endl;
  //  cout<<rewards<<endl;
  //  cout<<(actionSequences[maxRewardIdx].slice(1,discreteActions.size(),this->world.getTakenAction().size(),1) - savedCA.transpose(0,1)[maxRewardIdx])<<endl;;
  actionSequence = actionSequences[maxRewardIdx].to(torch::Device(torch::kCPU));   
  trajectory = stateSequences[maxRewardIdx].to(torch::Device(torch::kCPU));
  reward = -costs[maxRewardIdx].to(torch::Device(torch::kCPU));
}

template <class W, class F, class P>
void ModelBased<W,F,P>::playOne(ActionSpace actionSpace, int nRollouts, int nTimesteps, int nGradientSteps, float lr, torch::Tensor initActions)
{
  torch::Tensor sTruth = torch::zeros(0), sPred = torch::zeros(0);
  sTruth = torch::cat({sTruth,torch::tensor(this->currentState().getStateVector()).unsqueeze(0)},0);
  while(!this->world.isTerminal(this->currentState().getStateVector()))
    {
      gradientBasedPlanner(torch::tensor(this->currentState().getStateVector()), actionSpace,nRollouts,nTimesteps,nGradientSteps,lr, initActions);
      sPred = torch::cat({sPred,trajectory.slice(0,0,-1,1)},0);      
      for (int t=0;t<nTimesteps;t++)
	{
	  for (int i=0;i<this->world.getTakenAction().size();i++)
	    {
	      this->world.updateTakenAction(i,*actionSequence[t][i].data<float>());
	    }
	  this->world.transition();
	  sTruth = torch::cat({sTruth,torch::tensor(this->currentState().getStateVector()).unsqueeze(0)},0);  
	}
      cout<<this->rewardHistory()<<endl;
    }
  sPred = torch::cat({sPred,trajectory[-1].unsqueeze(0)},0);      
  //  cout<<torch::cat({sTruth.slice(1,0,2,1),b.slice(1,0,2,1)},1)<<endl;
  cout<<ToolsSS().moduloMSE(sPred.slice(1,0,2,1).slice(0,1,nTimesteps+1,1),sTruth.slice(1,0,2,1).slice(0,1,nTimesteps+1,1),false).pow(0.5)<<endl;
}


template <class W, class F, class P>
void ModelBased<W,F,P>::trainPolicyNetwork(torch::Tensor actionInputs, torch::Tensor stateInputs, int epochs, int batchSize, float lr)
{

  /*
  //Building demonstration dataset

  cout << "Building demonstration dataset..." << endl;
  torch::Tensor actions = torch::zeros(0);
  torch::Tensor states = torch::zeros(0);
  int n = batchSize*epochs;
  int m = actionInputs.size(0);

  
  
  //Training the planner

  vector<torch::Tensor> aBatches = torch::split(actions,batchSize,0);
  vector<torch::Tensor> sBatches = torch::split(states,batchSize,0);
  torch::optim::Adam optimizer(planner->parameters(),lr);
  
  for (int e=0;e<epochs;e++)
    {
      torch::Tensor aBatch = aBatches[e], sBatch = sBatch[e];
      torch::Tensor output = planner->forward(sBatch);
      torch::Tensor loss = torch::binary_cross_entropy(output,aBatch);
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      pLossHistory.push_back(*loss.to(torch::Device(torch::kCPU)).data<float>());

      if (e%25 == 0)
	{
	  cout<< "Training loss at epoch " + to_string(e)+"/"+to_string(epochs)+" : " + to_string(*loss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	}
    }
  */  
}

template <class W, class F, class P>
void ModelBased<W,F,P>::saveTrainingData()
{
  if (sLossHistory.size()!=0){
    ofstream s("../temp/transitionLoss");
    if (!s){cout<<"an error as occured while trying to save training data"<<endl;}
    for (unsigned int i=0;i<sLossHistory.size();i++)
    {
      s<<sLossHistory[i]<<endl;
    }
  }
  if (rLossHistory.size()!=0)
    {
      ofstream r("../temp/rewardLoss");
      if (!r){cout<<"an error as occured while trying to save training data"<<endl;}
      for (unsigned int i=0;i<rLossHistory.size();i++)
	{
	  r<<rLossHistory[i]<<endl;
	}
    }
}

template <class W, class F, class P>
F ModelBased<W,F,P>::getForwardModel()
{
  return forwardModel;
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::mergeDim(torch::Tensor x)
{
  int bs=x.size(0),T=x.size(1),c,s;
  if (x.dim()==2)
    {
      return x.reshape({bs*T});
    }
  else if (x.dim()>3)
    {
      c=x.size(2);
      s=x.size(3);
      return x.reshape({bs*T,c,s,s});
    }
  else
    {
      s=x.size(2);
      return x.reshape({bs*T,s});
    }
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::splitDim(torch::Tensor x, int bs, int T)
{
  int c,s;
  if (x.dim()==1)
    {
      return x.reshape({bs,T});
    }
  else if (x.dim()>2)
    {
      c=x.size(1);
      s=x.size(2);
      return x.reshape({bs,T,c,s,s});
    }
  else
    {
      s=x.size(1);
      return x.reshape({bs,T,s});
    }  
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::getInitTokens(vector<DiscreteAction> discreteActions, int T, int K)
{
  torch::Tensor initTokens = torch::zeros(0);
  for (unsigned int d=0;d<discreteActions.size();d++)
    {
      initTokens = torch::cat({initTokens,torch::zeros({T,K,discreteActions[d].getSize()}).normal_(0,1)},2);
    }
  return initTokens;
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::getInitCA(int nca, int T, int K)
{
  /*  torch::Tensor initCA = torch::zeros({K,nca,T});
  for (unsigned int i=0;i<nca;i++)
    {
      torch::Tensor center = torch::rand({K});
      for (int k=0;k<K;k++)
	{
	  initCA[k][i] = torch::zeros({T}).normal_(*center[k].data<float>(),0.1);
	}      
    }
    return torch::clamp(initCA,0,1).transpose(1,2).transpose(0,1); */
  return torch::rand({T,K,nca});
}

template <class W, class F, class P>
vector<torch::Tensor> ModelBased<W,F,P>::breakDAIntoTokens(vector<DiscreteAction> discreteActions, torch::Tensor toOptiDA)
{
  vector<torch::Tensor> daTokens;
  int sum=0;
  for (int d=0;d<discreteActions.size();d++)
    {
      int daSize = discreteActions[d].getSize();
      daTokens.push_back(toOptiDA.slice(2,sum,sum+daSize,1));
      sum+=daSize;
    }
  return daTokens;
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::tokensToOneHot(vector<torch::Tensor> daTokens, int daSize)
{
  torch::Tensor ohev = torch::zeros(0);
  for (int d=0;d<daSize;d++)
    {
      torch::Tensor toOneHot = torch::softmax(daTokens[d],-1);
      torch::Tensor maxToken = get<0>(torch::max(toOneHot,-1)).unsqueeze(2);
      toOneHot = (1/maxToken)*toOneHot - 0.9999;
      toOneHot = torch::relu(toOneHot) * 10000;
      ohev = torch::cat({ohev,toOneHot},-1);
    }
  return ohev;
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::getFinalDA(vector<DiscreteAction> discreteActions, torch::Tensor optiDA)
{
  torch::Tensor finalDA = torch::zeros(0); 
  int sum=0;
  for (int d=0;d<discreteActions.size();d++)
    {
      int daSize = discreteActions[d].getSize();
      finalDA = torch::cat({finalDA, get<1>(torch::max(optiDA.slice(2,sum,sum+daSize,1),2)).unsqueeze(2).to(torch::kFloat32)});
      sum+=daSize;
    }
  return finalDA;
}

template <class W, class F, class P>
torch::Tensor ModelBased<W,F,P>::getFinalCA(vector<ContinuousAction> continuousActions, torch::Tensor optiCA)
{
  torch::Tensor finalCA = optiCA.clone().transpose(0,-1);
  for (unsigned int c=0;c<continuousActions.size();c++)
    {
      float lb = continuousActions[c].getLowerBound();
      float ub = continuousActions[c].getUpperBound();      
      finalCA[c] = finalCA[c]*(ub-lb)+lb;
      //      savedCA[c] = savedCA[c]*(ub-lb)+lb;
    }
  return finalCA.transpose(0,-1);    

}

template <class W, class F, class P>
float ModelBased<W,F,P>::computeTrueReward(torch::Tensor initState, vector<DiscreteAction> discreteActions, torch::Tensor actions, vector<ContinuousAction> continuousActions, int nca, int nda, int nTimesteps)
{
  auto w(this->world);
  w.setCurrentState(State(ToolsSS().tensorToVector(initState)));
  w.generateVectorStates();
  torch::Tensor finalDA = getFinalDA(discreteActions, actions.slice(2,0,nda,1));
  torch::Tensor finalCA = getFinalCA(continuousActions, torch::clamp(actions.slice(2,nda,nda+nca,1),0,1));
  torch::Tensor actact = torch::cat({finalDA,finalCA},2);
  float r = 0;
  for (int u=0;u<nTimesteps;u++)
    {
      w.setTakenAction(ToolsSS().tensorToVector(actact[u][0]));
      r+=w.transition();
    }
}

template class ModelBased<GridWorld, ForwardGW, PlannerGW>;
template class ModelBased<SpaceWorld, ForwardSS, PlannerGW>;
