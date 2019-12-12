#include "modelbased.h"

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased():
  device(torch::Device(torch::kCPU))
{
}

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased(W world, F forwardModel):
  forwardModel(forwardModel), device(forwardModel->getUsedDevice())
{
  this->world = world;
}

template <class W, class F, class P>
ModelBased<W,F,P>::ModelBased(W world, F forwardModel, P planner):
  forwardModel(forwardModel),planner(planner),device(forwardModel->getUsedDevice())
{
  this->world = world;
}

template <class W, class F, class P>
void ModelBased<W,F,P>::learnForwardModel(torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize, float lr, float beta)
{
  int n = stateInputs.size(0);
  torch::optim::Adam optimizer(forwardModel->parameters(), lr);

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
	  siBatch = torch::cat({siBatch,stateInputs[index][0].unsqueeze(0)});
	  aiBatch = torch::cat({aiBatch,actionInputs[index].unsqueeze(0)});
	  slBatch = torch::cat({slBatch,stateLabels[index].unsqueeze(0)});
	  rlBatch[i] = rewardLabels[index]; 
	}

      siBatch = siBatch.to(device);
      aiBatch = aiBatch.to(device);
      slBatch = slBatch.to(device);
      rlBatch = rlBatch.to(device);
      
      //Forward and backward pass
      
      torch::Tensor stateOutputs = torch::zeros(0).to(device);
      torch::Tensor rewardOutputs = torch::zeros({nTimesteps,batchSize}).to(device);
      for (int t=0;t<nTimesteps;t++)
	{
	  forwardModel->forward(siBatch.squeeze(),aiBatch.transpose(0,1)[t]);
	  siBatch = forwardModel->predictedState;
	  stateOutputs = torch::cat({stateOutputs,siBatch.unsqueeze(1)},1);
	  rewardOutputs[t] = forwardModel->predictedReward;
	}
      torch::Tensor sLoss = beta*torch::mse_loss(stateOutputs,slBatch);
      torch::Tensor rLoss = torch::mse_loss(rewardOutputs.transpose(0,1),rlBatch); 
      torch::Tensor totalLoss = 100*(sLoss+rLoss);

      if (e == 5000)
	{
	  cout<<stateOutputs[0]<<endl;
	  cout<<slBatch[0]<<endl;
	  cout<<rewardOutputs.transpose(0,1)[0]<<endl;
	  cout<<rlBatch[0]<<endl;
	}
      optimizer.zero_grad();
      totalLoss.backward();
      optimizer.step();
      sLossHistory.push_back(*sLoss.to(torch::Device(torch::kCPU)).data<float>());
      rLossHistory.push_back(*rLoss.to(torch::Device(torch::kCPU)).data<float>());

      //Printing some stuff
      
      if (e%25 == 0)
	{
	  cout<< "Training loss at epoch " + to_string(e)+"/"+to_string(epochs)+" : " + to_string(*totalLoss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	}
    }  
}

template <class W, class F, class P>
void ModelBased<W,F,P>::gradientBasedPlanner(int nRollouts, int nTimesteps, int nGradsteps, float lr)
{
  torch::Tensor stateSequences = torch::zeros({nTimesteps+1,nRollouts,this->currentState().getStateVector().size()}); 
  torch::Tensor actionSequences = torch::zeros({nTimesteps,nRollouts,4});  
  torch::Tensor rewards = torch::zeros({nRollouts});
  torch::Tensor initState = torch::tensor(this->currentState().getStateVector());
  
  //  torch::Tensor tokens = torch::full({nTimesteps,nRollouts,4},0.1).to(torch::kFloat32);
  //  tokens[0][0][0]=40000;      

  torch::Tensor tokens = torch::zeros({nTimesteps,nRollouts,4}).normal_(0,0.1);
  tokens = torch::autograd::Variable(tokens.clone().set_requires_grad(true));       
  torch::Tensor qactionSequences = torch::softmax(tokens,2);
  for (int i=0;i<nGradsteps;i++)
    {
      for (int k=0;k<nRollouts;k++)
	{	  
	  stateSequences[0][k] = initState;
	}
      torch::Tensor totalReward = torch::zeros({nRollouts}).to(device);
      for (int t=0;t<nTimesteps;t++)
	{
	  //Clipping tokens to closest one-hot encoded vector
	  
	  torch::Tensor ohev = torch::zeros({nRollouts,4});	      	      
	  ohev = torch::softmax(tokens[t],1);
	  torch::Tensor maxToken = get<0>(torch::max(ohev,1)).unsqueeze(0).transpose(0,1);
	  ohev = (1/maxToken)*ohev - 0.999;
	  ohev = torch::relu(ohev) * 1000;
	  {
	    torch::NoGradGuard no_grad;		    
	    forwardModel->forward(stateSequences[t].to(device),ohev.to(device));
	    stateSequences[t+1]=forwardModel->predictedState;
	  }
	  forwardModel->forward(stateSequences[t].to(device),torch::softmax(tokens[t],1).to(device));
	  totalReward+=forwardModel->predictedReward;
	}
      //  cout<<100*totalReward.mean()<<endl;
      totalReward.backward(torch::ones({nRollouts}).to(device));
      rewards = totalReward;
      torch::Tensor grads = tokens.grad();
      torch::Tensor newTokens = tokens.clone().detach() + lr*grads;
      //cout<<torch::softmax(newTokens,1)-torch::softmax(tokens,1)<<endl;
      tokens = torch::autograd::Variable(newTokens.clone().detach().set_requires_grad(true));       
    }
  stateSequences = stateSequences.transpose(0,1);
  tokens = tokens.transpose(0,1);
  qactionSequences = qactionSequences.transpose(0,1);
  actionSequences = torch::softmax(tokens,2);
  int maxRewardIdx = *torch::argmax(rewards.to(torch::Device(torch::kCPU))).data<long>();
  cout<<"......"<<endl;
  cout<<qactionSequences[maxRewardIdx]<<endl;
  cout<<actionSequences[maxRewardIdx] - qactionSequences[maxRewardIdx]<<endl;
  cout<<actionSequences[maxRewardIdx]<<endl;      
  cout<<rewards[maxRewardIdx]<<endl;

  actionSequence = actionSequences[maxRewardIdx].to(torch::Device(torch::kCPU));
  trajectory = stateSequences[maxRewardIdx].to(torch::Device(torch::kCPU));
  reward = rewards[maxRewardIdx].to(torch::Device(torch::kCPU));
}

template <class W, class F, class P>
void ModelBased<W,F,P>::trainPolicyNetwork(torch::Tensor actionInputs, torch::Tensor stateInputs, int epochs, int batchSize, float lr)
{
  //Building demonstration dataset

  cout << "Building demonstration dataset..." << endl;

  torch::Tensor actions = torch::zeros(0);
  torch::Tensor states = torch::zeros(0);
  int n = batchSize*epochs;
  int m = actionInputs.size(0);
  for (int i=0;i<n/m;i++)
    {
      torch::Tensor perm = torch::randperm(m);
      actions = torch::cat({actions,actionInputs[perm]});
      states = torch::cat({states,stateInputs[perm]});
    }
  torch::Tensor perm = torch::randperm(m);
  actions = torch::cat({actions, torch::split(actionInputs[perm],n%m,1)[0]});
  states = torch::cat({states, torch::split(stateInputs[perm],n%m,1)[0]});
  
  //Training the planner

  vector<torch::Tensor> aBatches = torch::split(actions,batchSize,0);
  vector<torch::Tensor> sBatches = torch::split(states,batchSize,0);
  torch::optim::Adam optimizer(planner->parameters(),lr);
  
  for (int e=0;e<epochs;e++)
    {
      torch::Tensor aBatch = aBatches[e], sBatch = sBatch[e];
      torch::Tensor output = planner->forward(sBatch);
      torch::Tensor loss = torch::binary_cross_entropy(output,sBatch);
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }  
}


template <class W, class F, class P>
void ModelBased<W,F,P>::playOne(int nRollouts, int nTimesteps, int nGradientSteps, float lr)
{
  while(!this->world.isTerminal(this->currentState().getStateVector()))
    {
      gradientBasedPlanner(nRollouts,nTimesteps,nGradientSteps,lr);
      for (int t=0;t<nTimesteps;t++)
	{
	  this->world.setTakenAction(vector<float>({*torch::argmax(actionSequence[t]).data<long>()}));
	  this->world.transition();
	}
    }
  cout<<this->rewardHistory()<<endl;
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

template class ModelBased<GridWorld, ForwardGW, PlannerGW>;
template class ModelBased<SpaceWorld, ForwardSS, PlannerGW>;
