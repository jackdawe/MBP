#include "modelbased.h"

template <class W, class M, class P>
ModelBased<W,M,P>::ModelBased()
{
}

template <class W, class M, class P>
ModelBased<W,M,P>::ModelBased(W world, M model, P planner):
   model(model), planner(planner)
{
  this->world = world;
}

template<class W, class M, class P>
void ModelBased<W,M,P>::learnWorldModel(string path,int epochs, int batchSize, float lr)
{

  //Loading the dataset

  torch::Tensor stateInputs, actionInputs, stateLabels, rewardLabels;
  torch::load(stateInputs,path+"stateInputsTrain.pt");
  torch::load(actionInputs, path+"actionInputsTrain.pt");
  torch::load(stateLabels, path+"stateLabelsTrain.pt");
  torch::load(rewardLabels, path+"rewardLabelsTrain.pt");
  stateInputs = stateInputs.to(model->getUsedDevice());
  actionInputs = actionInputs.to(model->getUsedDevice());
  stateLabels = stateLabels.to(model->getUsedDevice());
  rewardLabels = rewardLabels.to(model->getUsedDevice());

  int m = stateInputs.size(0);
  int size = stateInputs.size(2);
  
  torch::optim::Adam optimizer(model->parameters(), lr);

  //Training Loop

  default_random_engine generator(random_device{}());
  uniform_int_distribution<int> dist(0,m-1);
  torch::Tensor loss;
  for (int e=0;e<epochs;e++)
    {
      //Extracting batch from dataset
    
      torch::Tensor siBatch = torch::zeros({batchSize,3,size,size}).to(model->getUsedDevice());
      torch::Tensor aiBatch = torch::zeros({batchSize}).to(model->getUsedDevice());
      torch::Tensor slBatch = torch::zeros({batchSize,size,size}).to(model->getUsedDevice());
      torch::Tensor rlBatch = torch::zeros({batchSize}).to(model->getUsedDevice());
      for (int i=0;i<batchSize;i++)
	{
	  int index = dist(generator);
	  siBatch[i] = stateInputs[index]; 
	  aiBatch[i] = actionInputs[index];
	  slBatch[i] = stateLabels[index];
	  rlBatch[i] = rewardLabels[index];
	}
      
      //Forward and backward pass
      torch::Tensor stateOutput = model->predictState(siBatch, aiBatch);
      torch::Tensor rewardOutput = model->predictReward(siBatch, aiBatch,true); 
      torch::Tensor rLoss = torch::nll_loss(rewardOutput, rlBatch.to(torch::kLong));
      torch::Tensor sLoss =  10*torch::binary_cross_entropy(stateOutput,slBatch); 
      torch::Tensor totalLoss = sLoss + rLoss;
      if (rLossHistory.size()!=0 && *totalLoss.to(torch::Device(torch::kCPU)).data<float>()-*loss.to(torch::Device(torch::kCPU)).data<float>() > 0.5)
	{
	  cout<<"High loss variation detected. Resuming training on last checkpoint..."<<endl;
	  torch::load(model,"../worldmodel-checkpoint.pt");
	  torch::load(optimizer,"../wmoptim-checkpoint.pt");	
	} else {
	loss = totalLoss;
	
	//Checkpointing 

	if (e%500 == 0 || e == 100)
	  {
	    torch::save(model,"../worldmodel-checkpoint.pt");
	    torch::save(optimizer,"../wmoptim-checkpoint.pt");
	    cout<<"Saving model and optimizer for checkpoint..."<<endl;
	  }
	    
	optimizer.zero_grad();
	totalLoss.backward();
	optimizer.step();
	rLossHistory.push_back(*rLoss.to(torch::Device(torch::kCPU)).data<float>());
	sLossHistory.push_back(*sLoss.to(torch::Device(torch::kCPU)).data<float>());
      }
      //Printing some stuff

      if (e%25 == 0)
	{
	  cout<< "Training loss at epoch " + to_string(e)+"/"+to_string(epochs)+" : " + to_string(*totalLoss.to(torch::Device(torch::kCPU)).data<float>())<<endl; 
	}
    } 
}

template <class W, class M, class P>
void ModelBased<W,M,P>::saveTrainingData(string dir)
{
  ofstream r(dir+"rewardLossWMGW");
  ofstream s(dir+"stateLossWMGW");

  if (!s)
    {
      cout<<"An error as occurred will trying to save the training data" << endl;
    }
  else
    {
      for (unsigned int i=0;i<rLossHistory.size();i++)
	{
	  r << rLossHistory[i] << endl;
	  s << sLossHistory[i] << endl;
	}
    }
}

template <class W, class M, class P>
M ModelBased<W,M,P>::getModel() const
{
  return model;
}

template class ModelBased<GridWorld, WorldModelGW, PlannerGW>;













































