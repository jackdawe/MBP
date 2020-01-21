#include "modelbased.h"
template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased():
  device(torch::Device(torch::kCPU))
{
}

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased(W world, F forwardModel):
  forwardModel(forwardModel), device(forwardModel->usedDevice)
{
  this->world = world;
}

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased(W world, F forwardModel, P planner):
  forwardModel(forwardModel),planner(planner),device(forwardModel->usedDevice)
{
  this->world = world;
}

template <class W, class F, class P>
void ModelBased<W,F,P>::learnForwardModel(torch::optim::Adam *optimizer, torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize, float beta, bool allStatesProvided)
{
  int n = stateInputs.size(0);
  int s;
  
  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
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
      siBatch = siBatch.to(device), aiBatch = aiBatch.to(device), slBatch = slBatch.to(device);
      rlBatch = rlBatch.to(device);
      
      //Forward and backward pass

      if (allStatesProvided)
	{
	  forwardModel->forward(mergeDim(siBatch),mergeDim(aiBatch));
	}
      else
	{
	  torch::Tensor stateOutputs, rewardOutputs;
	  stateOutputs = torch::zeros(siBatch.sizes()).transpose(0,1).to(device);
	  rewardOutputs = torch::zeros({nTimesteps,batchSize}).to(device);	  
	  forwardModel->forward(siBatch.transpose(0,1)[0],aiBatch.transpose(0,1)[0]);	      
	  stateOutputs[0] = forwardModel->predictedState;      
	  rewardOutputs[0] = forwardModel->predictedReward;
	  for (int t=1;t<nTimesteps;t++)
	    {
	      forwardModel->forward(stateOutputs[t-1],aiBatch.transpose(0,1)[t]);	      
	      stateOutputs[t] = forwardModel->predictedState;      
	      rewardOutputs[t] = forwardModel->predictedReward;
	    }
	  rewardOutputs = mergeDim(rewardOutputs.transpose(0,1));
	  stateOutputs = mergeDim(stateOutputs.transpose(0,1));
	  forwardModel->predictedState = stateOutputs; //Regrouping the operation for loss computation
	  forwardModel->predictedReward = rewardOutputs;
	}
      forwardModel->computeLoss(mergeDim(slBatch),mergeDim(rlBatch));
      torch::Tensor sLoss = beta*forwardModel->stateLoss, rLoss = forwardModel->rewardLoss;
      torch::Tensor totalLoss = sLoss+rLoss;
      optimizer->zero_grad();
      totalLoss.backward();
      optimizer->step();
      sLossHistory.push_back(*sLoss.to(torch::Device(torch::kCPU)).data<float>());
      rLossHistory.push_back(*rLoss.to(torch::Device(torch::kCPU)).data<float>());
      
      //Printing some stuff
      
      if (e%(epochs/25) == 0)
	{	  
	  cout<< "Training loss at iteration " + to_string(e)+"/"+to_string(epochs)+" : " + to_string(*totalLoss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	}
    }  
}

template <class W, class F, class P>
void ModelBased<W,F,P>::gradientBasedPlanner(ActionSpace actionSpace, int nRollouts, int nTimesteps, int nGradsteps, float lr)
{
  ofstream f("../temp/file1");

  //Setting everything up
  
  torch::Tensor initState = torch::tensor(this->currentState().getStateVector());
  unsigned int s = initState.size(0);

  torch::Tensor stateSequences = torch::zeros({nTimesteps+1,nRollouts,s});  
  torch::Tensor actionSequences = torch::zeros(0);
  torch::Tensor rewards = torch::zeros({nRollouts}).to(device);
  
  unsigned int nca = actionSpace.getContinuousActions().size();
  unsigned int nda = actionSpace.nActions()-nca;
  vector<DiscreteAction> discreteActions = actionSpace.getDiscreteActions();

  //Randomly initialising discrete and continuous actions and merging them

  torch::Tensor toOptimize = torch::zeros(0);  
  for (unsigned int d=0;d<discreteActions.size();d++)
    {
      toOptimize = torch::cat({toOptimize,torch::zeros({nTimesteps,nRollouts,discreteActions[d].getSize()}).normal_(0,1)},2);
    }

  for (unsigned int i=0;i<nca;i++)
    {
      torch::Tensor center = torch::rand({nRollouts});
      torch::Tensor initCA = torch::zeros({nRollouts,nTimesteps,1});
      for (int k=0;k<nRollouts;k++)
	{
	  initCA[k] = torch::clamp(torch::zeros({nTimesteps,1}).normal_(*center[k].data<float>(),0.1),0,1); 
	}      
      toOptimize = torch::cat({toOptimize,initCA.transpose(0,1)},2);
    }
  toOptimize = torch::autograd::Variable(toOptimize.clone().set_requires_grad(true));
  
  torch::Tensor savedCA;
  if (nca!=0) //check if necessary
    {
      savedCA=toOptimize.slice(2,nda,nda+nca);
    }
      
  //Looping over the number of optimisation steps 

  for (int i=0;i<nGradsteps+1;i++)
    {            
      for (int k=0;k<nRollouts;k++)
	{	  
	  stateSequences[0][k] = initState;
	}

      //Putting each discrete action into a vector and putting continuous actions into one tensor  
      
      torch::Tensor toOptiDA = toOptimize.slice(2,0,nda,1);
      torch::Tensor toOptiCA = torch::clamp(toOptimize.slice(2,nda,nda+nca,1),0,1);
      vector<torch::Tensor> daTokens;
      
      int sum=0;
      for (int d=0;d<discreteActions.size();d++)
	{
	  int daSize = discreteActions[d].getSize();
	  daTokens.push_back(toOptiDA.slice(2,sum,sum+daSize,1));
	  sum+=daSize;
	}

      //Predicting the state sequence given the initState and the action sequence

      for (int t=0;t<nTimesteps;t++)
	{
	  //Clipping tokens to closest one-hot encoded vector to minimise inference error
	  {
	    torch::NoGradGuard no_grad;
	    torch::Tensor ohev = torch::zeros(0);
	    for (int d=0;d<discreteActions.size();d++)
	      {
		torch::Tensor toOneHot = torch::softmax(daTokens[d][t],1);
		torch::Tensor maxToken = get<0>(torch::max(toOneHot,1)).unsqueeze(0).transpose(0,1);
		toOneHot = (1/maxToken)*toOneHot - 0.9999;
		toOneHot = torch::relu(toOneHot) * 10000;
	        ohev = torch::cat({ohev,toOneHot},1);
	      }
	    if (nca>0)
	      {
		ohev = torch::cat({ohev,toOptiCA[t]},1);
	      }
	    forwardModel->forward(stateSequences[t],ohev);
	    stateSequences[t+1]=forwardModel->predictedState;
	  }
	}
      //Predicting the rewards given the state and action sequences 

      if (i<nGradsteps)
	{      
	  torch::Tensor softmaxActions = torch::zeros(0);
	  for (int d=0;d<discreteActions.size();d++)
	    {
	      softmaxActions = torch::cat({softmaxActions,torch::softmax(daTokens[d],2)},2);
	    }      
	  softmaxActions = torch::cat({softmaxActions,toOptiCA},2);       
	  forwardModel->forward(torch::split(stateSequences,nTimesteps,0)[0].reshape({nTimesteps*nRollouts,s}),softmaxActions.reshape({nTimesteps*nRollouts,nca+nda}));
	  rewards = forwardModel->predictedReward.reshape({nTimesteps,nRollouts}).sum(0);	
	  //cout<< forwardModel->predictedReward.reshape({nTimesteps,nRollouts}) << endl;
	  //	  cout<<rewards.mean()<<endl;
	  rewards.backward(torch::ones({nRollouts}).to(device));
	  torch::Tensor grads = toOptimize.grad();
	  //      cout<<lr*grads.transpose(0,1)[0]<<endl;
	  torch::Tensor optiActions = toOptimize.clone().detach() + lr*grads;      
	  
	  //Updating some tensors with the new action values 
	  
	  toOptimize = torch::autograd::Variable(optiActions.clone().detach().set_requires_grad(true));
	}
    }
  stateSequences = stateSequences.transpose(0,1);
  toOptimize = toOptimize.transpose(0,1);
  torch::Tensor optiDA = toOptimize.slice(2,0,nda,1);
  torch::Tensor optiCA = toOptimize.slice(2,nda,nda+nca,1);
  int sum=0;
  for (int d=0;d<discreteActions.size();d++)
    {
      int daSize = discreteActions[d].getSize();
      actionSequences = torch::cat({actionSequences, get<1>(torch::max(optiDA.slice(2,sum,sum+daSize,1),2)).unsqueeze(2).to(torch::kFloat32)});
      sum+=daSize;
    }

  //Putting back continuous actions within their bounds

  optiCA = torch::clamp(optiCA.transpose(0,-1),0,1);
  for (int c=0;c<nca;c++)
    {
      float lb = actionSpace.getContinuousActions()[c].getLowerBound();
      float ub = actionSpace.getContinuousActions()[c].getUpperBound();      
      optiCA[c] = optiCA[c]*(ub-lb)+lb;
      savedCA[c] = savedCA[c]*(ub-lb)+lb;
    }
  optiCA = optiCA.transpose(0,-1);    
  if (nca>0)
    {
      actionSequences = torch::cat({actionSequences,optiCA},2);
    }
  int maxRewardIdx = *torch::argmax(rewards.to(torch::Device(torch::kCPU))).data<long>();
  cout<<"......"<<endl;
  //cout<<actionSequences[maxRewardIdx]<<endl;      
  cout<<rewards[maxRewardIdx]<<endl;
  //  cout<<rewards<<endl;
  //  cout<<(actionSequences[maxRewardIdx].slice(1,discreteActions.size(),this->world.getTakenAction().size(),1) - savedCA.transpose(0,1)[maxRewardIdx])<<endl;;
  
  actionSequence = actionSequences[maxRewardIdx].to(torch::Device(torch::kCPU));   
  trajectory = stateSequences[maxRewardIdx].to(torch::Device(torch::kCPU));
  reward = rewards[maxRewardIdx].to(torch::Device(torch::kCPU));
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
void ModelBased<W,F,P>::playOne(ActionSpace actionSpace, int nRollouts, int nTimesteps, int nGradientSteps, float lr)
{
  torch::Tensor a = torch::zeros(0);
  torch::Tensor b = torch::zeros(0);
  a = torch::cat({a,torch::tensor(this->currentState().getStateVector()).unsqueeze(0)},0);
  while(!this->world.isTerminal(this->currentState().getStateVector()))
    {
      gradientBasedPlanner(actionSpace,nRollouts,nTimesteps,nGradientSteps,lr);      
      b = torch::cat({b,trajectory.slice(0,0,-1,1)},0);      
      for (int t=0;t<nTimesteps;t++)
	{
	  for (int i=0;i<this->world.getTakenAction().size();i++)
	    {
	      this->world.updateTakenAction(i,*actionSequence[t][i].data<float>());
	    }
	  this->world.transition();
	  a = torch::cat({a,torch::tensor(this->currentState().getStateVector()).unsqueeze(0)},0);  
	}
      cout<<this->rewardHistory()<<endl;
    }
  b = torch::cat({b,trajectory[-1].unsqueeze(0)},0);      
  cout<<torch::cat({a.slice(1,0,2,1),b.slice(1,0,2,1)},1)<<endl;
  cout<<ToolsSS().moduloMSE(a.slice(1,0,2,1),b.slice(1,0,2,1),false).pow(0.5)<<endl;
  /*
  torch::Tensor aaa = actionSequence.slice(1,1,3,1);
  torch::Tensor bb = actionSequence.slice(1,0,1,1);
  torch::Tensor bbb = torch::zeros({nTimesteps,4});
  for (int i=0;i<nTimesteps;i++)
    {
      bbb[i][(int)(*bb[i].data<float>())]=1;
    }

  torch::Tensor act = ToolsSS().normalizeActions(torch::cat({bbb,aaa},1).unsqueeze(0));
  torch::Tensor predictedStates = torch::zeros({nTimesteps,1,17});
  torch::Tensor predictedRewards = torch::zeros({nTimesteps,1});
  forwardModel->forward(b.unsqueeze(0).transpose(0,1)[0],act.transpose(0,1)[0]);
  predictedStates[0] = forwardModel->predictedState;      
  predictedRewards[0] = forwardModel->predictedReward;
  for (int t=1;t<nTimesteps;t++)
    {
      forwardModel->forward(predictedStates[t-1],act.transpose(0,1)[t]);
      predictedStates[t] = forwardModel->predictedState;      
      predictedRewards[t] = forwardModel->predictedReward;		    
    }
  forwardModel->predictedState = predictedStates.transpose(0,1).squeeze();
  forwardModel->predictedReward = predictedRewards.transpose(0,1);
  cout<<ToolsSS().moduloMSE(b.slice(1,0,2,1).slice(0,1,nTimesteps+1,1),forwardModel->predictedState.slice(1,0,2,1)).pow(0.5)<<endl;
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
vector<float> ModelBased<W,F,P>::tensorToVector(torch::Tensor stateVector)
{
  vector<float> vec;
  for (int i=0;i<stateVector.size(0);i++)
    {
      vec.push_back(*stateVector[i].data<float>());
    }
  return vec;
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

template class ModelBased<GridWorld, ForwardGW, PlannerGW>;
template class ModelBased<SpaceWorld, ForwardSS, PlannerGW>;
