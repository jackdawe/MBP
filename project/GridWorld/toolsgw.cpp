#include "toolsgw.h"

ToolsGW::ToolsGW():
  tScores(0), rScores(vector<int>(4,0)), rCounts(vector<int>(4,0)), tMSE(torch::zeros({1})), rMSE(torch::zeros({1})){}

ToolsGW::ToolsGW(GridWorld gw):
  gw(gw), tScores(0), rScores(vector<int>(4,0)), rCounts(vector<int>(4,0)), tMSE(torch::zeros({1})), rMSE(torch::zeros({1})){}

vector<float> ToolsGW::tensorToVector(torch::Tensor stateVector)
{
  vector<float> vec;
  for (int i=0;i<stateVector.size(0);i++)
    {
      vec.push_back(*stateVector[i].data<float>());
    }
  return vec;
}

torch::Tensor ToolsGW::toRGBTensor(torch::Tensor batch)
{
  int n = batch.size(0), T = batch.size(1), size = sqrt(batch.size(-1)-4);
  torch::Tensor rChannel = posToChannel(batch.slice(-1,0,2,1),size,n,T).unsqueeze(-3);
  torch::Tensor gChannel = posToChannel(batch.slice(-1,2,4,1),size,n,T).unsqueeze(-3);
  torch::Tensor bChannel = batch.slice(-1,4,batch.size(-1),1).reshape({n,T,1,size,size});
  return torch::cat({rChannel,gChannel,bChannel},-3);
}

torch::Tensor ToolsGW::posToChannel(torch::Tensor pos, int chanSize, int batchDim1, int batchDim2)
{
  torch::Tensor zeros = torch::zeros({chanSize,batchDim1,batchDim2});
  pos = pos.transpose(0,2).transpose(1,2);
  torch::Tensor x = zeros.clone().scatter_(0,pos.slice(0,0,1,1).to(torch::kLong),torch::ones_like(zeros)).transpose(0,2).transpose(0,1).unsqueeze(-1);
  torch::Tensor y = zeros.clone().scatter_(0,pos.slice(0,1,2,1).to(torch::kLong),torch::ones_like(zeros)).transpose(0,2).transpose(0,1).unsqueeze(-1).transpose(-1,-2);
  return torch::matmul(x,y);
}

torch::Tensor ToolsGW::generateActions(int n, int nTimesteps)
{
  torch::Tensor direction = torch::zeros({4,nTimesteps,n});
  direction = direction.scatter_(0,torch::randint(0,4,{1,nTimesteps,n}).to(torch::kLong),torch::ones_like(direction)).transpose(0,2);  
  return direction;
}

void ToolsGW::generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp)
{
  cout<<"Generating a dataset for the GridWorld task containing " + to_string(n) + " samples of " + to_string(nTimesteps) + " time steps.";
  cout<<"The training set contains " +to_string((int)(100*trainSetProp))+"% of the dataset and the test set the remaining samples."<<endl;
  cout<<"To help with the training, the agent is forced to make a transition towards the goal in " + to_string((int)(100*winProp)) + "% of the episodes."<<endl;

  gw = GridWorld(path+"train/",nmaps);  
  int nTr=(int)(trainSetProp*n), nTe = n-nTr;
  
  //Initialising the tensors that will contain the training set

  int size = gw.getSize();
  torch::Tensor stateInputs = torch::zeros({n,nTimesteps,size*size+4});
  torch::Tensor actionInputs = generateActions(n,nTimesteps);
  torch::Tensor stateLabels = torch::zeros({n,nTimesteps,size*size+4});
  torch::Tensor rewardLabels = torch::zeros({n,nTimesteps});

  //Generating another action tensor where signal is encoded as an int

  torch::Tensor ieActions = torch::argmax(actionInputs,-1).unsqueeze(-1).to(torch::kFloat32);
  
  //Making the agent wander randomly for n episodes 
  
  int i=0;
  bool dispPerc = true;
  while (i<n)
    {      
      //Displaying a progression bar in the terminal

      if (dispPerc && n > 100 && i%(n/100) == 0)
	{
	  cout << "Your agent is crashing into planets for science... " + to_string((int)(i/(n/100.))) + "%" << endl;
	  dispPerc = false;
	}	  

     //Swapping to test set generation when training set generation is done      
      
      if (i==nTr)
	{
	  gw = GridWorld(path+"test/",nmaps);
	  cout<< "Training set generation is complete! Now generating test set..."<<endl; 
	}
      bool hitsGoal = false;
      for (int t=0;t<nTimesteps;t++)
	{      
	  //Building the dataset tensors
      
	  stateInputs[i][t] = torch::tensor(gw.getCurrentState().getStateVector());
	  float r = gw.transition(tensorToVector(ieActions[i][t]));
	  rewardLabels[i][t] = r;
	  if (r == WIN_REWARD)
	    {
	      hitsGoal = true;
	    }
	  stateLabels[i][t] = torch::tensor(gw.getCurrentState().getStateVector());
	}
      if ((i>=winProp*nTr && i<=n-winProp*nTe) || hitsGoal)
	{
	  i++;
	  dispPerc = true;
	  hitsGoal = false;	  
	}
      gw.reset();                  
    }
  
  //Saving the test set
  
  cout<< "Test set generation is complete!"<<endl;
  stateInputs = toRGBTensor(stateInputs);
  stateLabels = toRGBTensor(stateLabels);
  torch::save(stateInputs.slice(0,0,nTr,1),path+"stateInputsTrain.pt");
  torch::save(actionInputs.slice(0,0,nTr,1),path+"actionInputsTrain.pt");
  torch::save(rewardLabels.slice(0,0,nTr,1),path+"rewardLabelsTrain.pt");
  torch::save(stateLabels.slice(0,0,nTr,1),path+"stateLabelsTrain.pt");    
  torch::save(stateInputs.slice(0,nTr,nTr+nTe,1),path+"stateInputsTest.pt");
  torch::save(actionInputs.slice(0,nTr,nTr+nTe,1),path+"actionInputsTest.pt");
  torch::save(rewardLabels.slice(0,nTr,nTr+nTe,1),path+"rewardLabelsTest.pt");
  torch::save(stateLabels.slice(0,nTr,nTr+nTe,1),path+"stateLabelsTest.pt");  
}

void ToolsGW::transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit)
{
  tMSE = torch::mse_loss(testData,labels)/nSplit;
  testData = testData.to(torch::Device(torch::kCPU));
  labels = labels.to(torch::Device(torch::kCPU));  
  testData = torch::round(testData);
  tScores = *torch::eq(labels,testData).sum().data<long>();
}

void ToolsGW::displayTAccuracy(int dataSetSize)
{
  cout<<"\n########## TRANSITION FUNCTION EVALUATION ##########\n"<<endl;
  cout<<"Correct state images: " + to_string(tScores)+"/"+to_string(dataSetSize) + " (" + to_string(100.*tScores/dataSetSize) + "%)" << endl;
  cout<<endl;
  cout<< "PIXEL WISE AVERAGE MSE: ";
  cout<<pow(*tMSE.data<float>(),0.5)<<endl;
  cout<<endl;  
  cout<<"################################################"<<endl;  
}

void ToolsGW::rewardAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit)
{
  rMSE+=torch::mse_loss(testData,labels)/nSplit;
  testData = testData.to(torch::Device(torch::kCPU));
  labels = labels.to(torch::Device(torch::kCPU));    
  torch::Tensor precision = torch::abs(testData-labels);
  vector<float> thresholds = {0.25,0.25,0.1,0.1};
  vector<float> rewards = {WIN_REWARD,LOSE_REWARD,EMPTY_SQUARE_REWARD,0};
  for (int i=0;i<4;i++)
    {
      torch::Tensor forCounting = torch::eq(labels,torch::full(labels.sizes(),rewards[i]));
      rCounts[i] += *forCounting.sum().data<long>();
      rScores[i] += *(torch::lt(precision,torch::full(labels.sizes(),thresholds[i])*forCounting)).sum().data<long>();
    }  
}

void ToolsGW::displayRAccuracy()
{
  vector<string> text = {"WIN","LOSE","EMPTY SQUARE","PADDING"};
  cout<<"\n########## REWARD FUNCTION EVALUATION ##########\n " << endl;
  for (int i=0;i<4;i++)
    {
      cout<<text[i]+": "+ to_string(rScores[i]) + "/" + to_string(rCounts[i]) + " (" + to_string(100.*rScores[i]/rCounts[i]) + "%)"<<endl;
    }
  cout<<endl;
  cout<< "REWARD AVERAGE ERROR : ";
  cout<<pow(*rMSE.data<float>(),0.5)<<endl;
  cout<<endl;
  cout<<"################################################"<<endl;
}
