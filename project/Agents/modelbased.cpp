#include "modelbased.h"

template <class W, class T, class R, class P>
ModelBased<W,T,R,P>::ModelBased()
{
}

template <class W, class T, class R, class P>
ModelBased<W,T,R,P>::ModelBased(W world, T transitionFunction):
  transitionFunction(transitionFunction)
{
  this->world = world;
}

template <class W, class T, class R, class P>
ModelBased<W,T,R,P>::ModelBased(W world, R rewardFunction):
  rewardFunction(rewardFunction)
{
  this->world = world;
}

template <class W, class T, class R, class P>
ModelBased<W,T,R,P>::ModelBased(W world, T transitionFunction, R rewardFunction, P planner):
  transitionFunction(transitionFunction), rewardFunction(rewardFunction),planner(planner)
{
  this->world = world;
}

template <class W, class T, class R, class P>
void ModelBased<W,T,R,P>::learnTransitionFunction(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor labels, int epochs, int batchSize, float lr)
{
   //Migrating tensors to CUDA if available 

  stateInputs = stateInputs.to(transitionFunction->getUsedDevice());
  actionInputs = actionInputs.to(transitionFunction->getUsedDevice());
  labels = labels.to(transitionFunction->getUsedDevice());

  int n = stateInputs.size(0);
  int imSize = stateInputs.size(2);
  torch::optim::Adam optimizer(transitionFunction->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset
    
      torch::Tensor siBatch = torch::zeros({batchSize,3,imSize,imSize}).to(transitionFunction->getUsedDevice());
      torch::Tensor aiBatch = torch::zeros({batchSize,4}).to(transitionFunction->getUsedDevice());
      torch::Tensor lBatch = torch::zeros({batchSize,imSize,imSize}).to(transitionFunction->getUsedDevice());
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch[i] = stateInputs[index]; 
	  aiBatch[i] = actionInputs[index];
	  lBatch[i] = labels[index];
	}
      
      //Forward and backward pass

      torch::Tensor output = transitionFunction->predictState(siBatch, aiBatch); 
      torch::Tensor loss =  torch::binary_cross_entropy(output,lBatch); 
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      sLossHistory.push_back(*loss.to(torch::Device(torch::kCPU)).data<float>());
      
      //Printing some stuff
      
      if (e%25 == 0)
	{
	  cout<< "Training loss at epoch " + to_string(e)+"/"+to_string(epochs)+" : " + to_string(*loss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	}
    }  
}

template <class W, class T, class R, class P>
void ModelBased<W,T,R,P>::learnRewardFunction(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor labels, int epochs, int batchSize, float lr)
{
   //Migrating tensors to CUDA if available 

  stateInputs = stateInputs.to(rewardFunction->getUsedDevice());
  actionInputs = actionInputs.to(rewardFunction->getUsedDevice());
  labels = labels.to(rewardFunction->getUsedDevice());

  int n = stateInputs.size(0);
  int imSize = stateInputs.size(2);
  torch::optim::Adam optimizer(rewardFunction->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset
    
      torch::Tensor siBatch = torch::zeros({batchSize,3,imSize,imSize}).to(rewardFunction->getUsedDevice());
      torch::Tensor aiBatch = torch::zeros({batchSize,4}).to(rewardFunction->getUsedDevice());
      torch::Tensor lBatch = torch::zeros({batchSize}).to(rewardFunction->getUsedDevice());
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch[i] = stateInputs[index]; 
	  aiBatch[i] = actionInputs[index];
	  lBatch[i] = labels[index];
	}
      
      //Forward and backward pass

      torch::Tensor output = rewardFunction->predictReward(siBatch, aiBatch);
      torch::Tensor loss =  torch::nll_loss(output,lBatch.to(torch::kLong)); 
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      rLossHistory.push_back(*loss.to(torch::Device(torch::kCPU)).data<float>());
      
      //Printing some stuff
      
      if (e%25 == 0)
	{
	  cout<< "Training loss at epoch " + to_string(e)+"/"+to_string(epochs)+" : " + to_string(*loss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	}
    }  
}

template <class W, class T, class R, class P>
void ModelBased<W,T,R,P>::gradientBasedPlanner(int nRollouts, int nTimesteps, int nGradsteps, float lr)
{
  torch::Tensor stateSequences = torch::zeros({nRollouts,nTimesteps,3,8,8}); //MEH
  torch::Tensor actionSequences = torch::zeros({nRollouts,nTimesteps});
  torch::Tensor rewards = torch::zeros({nRollouts});
  
  for (int k=0;k<nRollouts;k++)
    {
      stateSequences[k][0] = this->world.toRGBTensor(this->currentState().getStateVector());
      torch::Tensor tokens = torch::randn({nRollouts});
      for (int i=0;i<nGradsteps;i++)
	{
	  actionSequences[k] = torch::softmax(tokens,0);
	  torch::Tensor r = torch::zeros({1},torch::requires_grad());
	  for (int t=0;t<nTimesteps;t++)
	    {
	      stateSequences[k][t+1] = transitionFunction->predictState(stateSequences[k][t].to(transitionFunction->getUsedDevice()),actionSequences[k][t].to(transitionFunction->getUsedDevice()));
	      r+=rewardFunction->predictReward(stateSequences[k][t].to(transitionFunction->getUsedDevice()),actionSequences[k][t].to(transitionFunction->getUsedDevice()));
	    }
	  rewards[k] = r;
	  r.backward();
	  for (int t=nTimesteps-1;t>0;t--)
	    {
	      tokens[t]-= lr*tokens[t].grad();
	    }
	  actionSequences[k] = torch::softmax(tokens,0);	 
	}
    }
  int maxRewardIdx = *torch::argmax(rewards).data<long>();
  cout<<rewards<<endl;
}

template <class W, class T, class R, class P>
void ModelBased<W,T,R,P>::saveTrainingData()
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

template <class W, class T, class R, class P>
T ModelBased<W,T,R,P>::getTransitionFunction()
{
  return transitionFunction;
}

template <class W, class T, class R, class P>
R ModelBased<W,T,R,P>::getRewardFunction()
{
  return rewardFunction;
}


template class ModelBased<GridWorld, TransitionGW, RewardGW, PlannerGW>;
