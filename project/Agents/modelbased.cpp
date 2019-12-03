#include "modelbased.h"
DEFINE_int32(K,10,"number of rollouts");
DEFINE_int32(T,10,"number of steps to plan");
DEFINE_int32(gs,10,"number of gradient steps");

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

  torch::Tensor qactionSequences = torch::zeros({nRollouts,nTimesteps,4});

  for (int k=0;k<nRollouts;k++)
    {
      torch::Tensor tokens = torch::full({nTimesteps,4},0.1).to(torch::kFloat32);
      tokens[0][0]=0.5,tokens[0][1]=0.4999;
      tokens = torch::autograd::Variable(tokens.clone().detach().set_requires_grad(true));

      //torch::Tensor tokens = torch::randn({nTimesteps,4},torch::requires_grad());
      
      qactionSequences[k] = torch::softmax(tokens,1);
      for (int i=0;i<nGradsteps;i++)
	{
	  stateSequences[k][0] = initState;
	  torch::Tensor totalReward = torch::zeros({1});
 	  for (int t=0;t<nTimesteps;t++)	    
	    {
	      //Clipping tokens to closest one-hot encoded vector

	      torch::Tensor ohev = torch::zeros({4});	      	      
	      ohev = torch::softmax(tokens[t],0);
	      ohev = (1/torch::max(ohev))*ohev - 0.999;
	      ohev = torch::relu(ohev) * 1000;
	      actionSequences[k][t] = ohev;		
	      if (this->world.isTerminal(State(tensorToVector(stateSequences[k][t]))))
		{
		  stateSequences[k][t+1] = stateSequences[k][t];
		}
	      else
		{
		  {
		    torch::NoGradGuard no_grad;
		    stateSequences[k][t+1] = transitionFunction->predictState(stateSequences[k][t].unsqueeze(0),ohev.unsqueeze(0).to(transitionFunction->getUsedDevice()))[0];
		  }
		  //torch::Tensor softToken = torch::softmax(tokens[t],0).unsqueeze(0).to(transitionFunction->getUsedDevice());
		  totalReward += rewardFunction->predictReward(stateSequences[k][t].unsqueeze(0),ohev.unsqueeze(0).to(transitionFunction->getUsedDevice()))[0].to(torch::Device(torch::kCPU));
		}
	    }
	  totalReward.backward();
	  rewards[k] = totalReward[0];
	  torch::Tensor grads = tokens.grad();
	  cout<<grads<<endl;
	  torch::Tensor newTokens = tokens.clone().detach() + lr*grads;
	  //cout<<torch::softmax(newTokens,1)-torch::softmax(tokens,1)<<endl;
	  tokens = torch::autograd::Variable(newTokens.clone().detach().set_requires_grad(true));
	}
      actionSequences[k] = torch::softmax(tokens,1);
    }
  int maxRewardIdx = *torch::argmax(rewards).data<long>();
  cout<<"......"<<endl;
  cout<<actionSequences[maxRewardIdx] - qactionSequences[maxRewardIdx]<<endl;
  cout<<actionSequences[maxRewardIdx]<<endl;      
  cout<<rewards[maxRewardIdx]<<endl;
  for (int o=0;o<6;o++)
    {
      //      cout<<ToolsGW().toRGBTensor(stateSequences[maxRewardIdx])[o][0]<<endl;
    }
  //cout<<rewards<<endl;
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
