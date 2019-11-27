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
  cout<<"1"<<endl;
  ToolsGW().toRGBTensor(stateInputs);
  cout<<"2"<<endl;
  int n = stateInputs.size(0);
  int svecSize = stateInputs.size(1);
  torch::optim::Adam optimizer(transitionFunction->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset
    
      torch::Tensor siBatch = torch::zeros({batchSize,svecSize});
      torch::Tensor aiBatch = torch::zeros(0);
      torch::Tensor lBatch = torch::zeros(0);
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch[i] = stateInputs[index]; 
	  aiBatch = torch::cat({aiBatch,actionInputs[index].unsqueeze(0)});
	  lBatch = torch::cat({lBatch,labels[index].unsqueeze(0)});
	}
      
      aiBatch = aiBatch.to(transitionFunction->getUsedDevice());
      lBatch = lBatch.to(transitionFunction->getUsedDevice());

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
  int n = stateInputs.size(0);
  int svecSize = stateInputs.size(1);
  torch::optim::Adam optimizer(rewardFunction->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset
    
      torch::Tensor siBatch = torch::zeros({batchSize,svecSize}).to(rewardFunction->getUsedDevice());
      torch::Tensor aiBatch = torch::zeros(0).to(rewardFunction->getUsedDevice());
      torch::Tensor lBatch = torch::zeros({batchSize}).to(rewardFunction->getUsedDevice());
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch[i] = stateInputs[index]; 
	  aiBatch = torch::cat({aiBatch,actionInputs[index]},0);
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
  /*  torch::Tensor stateSequences = torch::zeros({nRollouts,nTimesteps+1,8,8}); //MEH
  torch::Tensor actionSequences = torch::zeros({nRollouts,nTimesteps,4});
  torch::Tensor rewards = torch::zeros({nRollouts});
  torch::Tensor initState = this->world.toRGBTensor(this->currentState().getStateVector());
  
  for (int k=0;k<nRollouts;k++)
    {
      stateSequences[k][0] = initState[0];
      torch::Tensor tokens = torch::randn({nTimesteps,4},torch::requires_grad());
      for (int i=0;i<nGradsteps;i++)
	{
	  actionSequences[k] = torch::softmax(tokens,1);
	  torch::Tensor stepRewards = torch::zeros({nTimesteps});
 	  for (int t=0;t<nTimesteps;t++)
	    {
	      torch::Tensor rgbState = torch::zeros({1,3,8,8}).to(transitionFunction->getUsedDevice()); 
	      rgbState[0][0] = stateSequences[k][t], rgbState[0][1] = initState[1], rgbState[0][2] = initState[2];
	      if (true)
		{
		  stateSequences[k][t+1] = stateSequences[k][t];
		  stepRewards[t] = 0;
		}
	      else
		{
		  stateSequences[k][t+1] = transitionFunction->predictState(rgbState,softmax(tokens[t].reshape({1,4}),1).to(transitionFunction->getUsedDevice()))[0];
		  torch::Tensor rLogProbs = rewardFunction->predictReward(rgbState,actionSequences[k][t].reshape({1,4}).to(transitionFunction->getUsedDevice()));
		  int rId = *torch::argmax(torch::exp(rLogProbs)).to(torch::Device(torch::kCPU)).data<long>();
		  stepRewards[t]=this->world.idToReward(rId);
		}
	    }
	  cout<<torch::round(stateSequences[k])<<endl;
	  torch::Tensor totalStepReward = stepRewards.sum();
	  rewards[k] = totalStepReward;
	  totalStepReward.backward();
	  for (int t=nTimesteps-1;t>0;t--)
	    {
	      tokens[t]-= lr*tokens[t].grad();
	    }
	  actionSequences[k] = torch::softmax(tokens,1);	 
	}
    }
  int maxRewardIdx = *torch::argmax(rewards).data<long>();
  cout<<rewards<<endl;*/
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
