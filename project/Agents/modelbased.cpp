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
  int n = stateInputs.size(0);
  torch::optim::Adam optimizer(transitionFunction->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset
    
      torch::Tensor siBatch = torch::zeros(0);
      torch::Tensor aiBatch = torch::zeros(0);
      torch::Tensor lBatch = torch::zeros(0);
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch = torch::cat({siBatch,stateInputs[index].unsqueeze(0)});
	  aiBatch = torch::cat({aiBatch,actionInputs[index].unsqueeze(0)});
	  lBatch = torch::cat({lBatch,labels[index].unsqueeze(0)});
	}

      siBatch = siBatch.to(transitionFunction->getUsedDevice());
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
  torch::optim::Adam optimizer(rewardFunction->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,n-1);
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset

      torch::Tensor siBatch = torch::zeros(0);
      torch::Tensor aiBatch = torch::zeros(0);      
      torch::Tensor lBatch = torch::zeros({batchSize}).to(rewardFunction->getUsedDevice());
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch = torch::cat({siBatch,stateInputs[index].unsqueeze(0)});
	  aiBatch = torch::cat({aiBatch,actionInputs[index].unsqueeze(0)});
	  lBatch[i] = labels[index];
	}

      siBatch = siBatch.to(transitionFunction->getUsedDevice());
      aiBatch = aiBatch.to(transitionFunction->getUsedDevice());
      
      //Forward and backward pass

      torch::Tensor output = rewardFunction->predictReward(siBatch, aiBatch);
      torch::Tensor loss =  torch::mse_loss(output.squeeze(),lBatch); 
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
  torch::Tensor stateSequences = torch::zeros({nRollouts,nTimesteps+1,this->currentState().getStateVector().size()}); 
  torch::Tensor actionSequences = torch::zeros({nRollouts,nTimesteps,4});
  torch::Tensor rewards = torch::zeros({nRollouts});
  torch::Tensor initState = torch::tensor(this->currentState().getStateVector());
  
  for (int k=0;k<nRollouts;k++)
    {
      stateSequences[k][0] = initState;
      torch::Tensor tokens = torch::randn({nTimesteps,4},torch::requires_grad());
      for (int i=0;i<nGradsteps;i++)
	{
	  actionSequences[k] = torch::softmax(tokens,1);
	  torch::Tensor stepRewards = torch::zeros({nTimesteps});
 	  for (int t=0;t<nTimesteps;t++)
	    {
	      if (this->world.isTerminal(State(tensorToVector(stateSequences[k][t]))))
		{
		  stateSequences[k][t+1] = stateSequences[k][t];
		  stepRewards[t] = 0;
		}
	      else
		{
		  stateSequences[k][t+1] = transitionFunction->predictState(stateSequences[k][t].unsqueeze(0),softmax(tokens[t].unsqueeze(0),1).to(transitionFunction->getUsedDevice()))[0];
		  torch::Tensor rLogProbs = rewardFunction->predictReward(stateSequences[k][t].unsqueeze(0),actionSequences[k][t].reshape({1,4}).to(transitionFunction->getUsedDevice()));
		  torch::Tensor q = torch::argmax(torch::exp(rLogProbs));
		  int rId = *torch::argmax(torch::exp(rLogProbs)).to(torch::Device(torch::kCPU)).data<long>();
		  rLogProbs.backward();
		  cout<<"hey"<<endl;
		  stepRewards[t]=this->world.idToReward(rId);
		}
	    }
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

template <class W, class T, class R, class P>
vector<float> ModelBased<W,T,R,P>::tensorToVector(torch::Tensor stateVector)
{
  vector<float> vec;
  for (int i=0;i<stateVector.size(0);i++)
    {
      vec.push_back(*stateVector[i].data<float>());
    }
  return vec;
}

template class ModelBased<GridWorld, TransitionGW, RewardGW, PlannerGW>;
