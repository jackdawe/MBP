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
  
  int size = sqrt(batch.size(1)-4);
  torch::Tensor rgbState = torch::zeros({batch.size(0),3,size,size});
  for (int s=0;s<batch.size(0);s++)
    {
      for (int i=0;i<size;i++)
	{
	  for (int j=0;j<size;j++)
	    {
	      rgbState[s][2][i][j] = batch[s][i*size+j+4];
	    }
	}
      rgbState[s][0][(int)*batch[s][0].data<float>()][(int)*batch[s][1].data<float>()] = 1;
      rgbState[s][1][(int)*batch[s][2].data<float>()][(int)*batch[s][3].data<float>()] = 1;
    }
  return rgbState;
}

/*
cv::Mat ToolsGW::toRGBMat(torch::Tensor batch)
{
    cv::Mat rgbState(size,size,CV_8UC3);
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            for (int k=0;k<3;k++)
            {
                rgbState.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
        }
    }

    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            if (stateVector[i*size+j+4] == 1)
            {
                rgbState.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,255);
            }
        }
    }

    rgbState.at<cv::Vec3b>(stateVector[0],stateVector[1]) += cv::Vec3b(255,0,0);
    rgbState.at<cv::Vec3b>(stateVector[2],stateVector[3]) += cv::Vec3b(0,255,0);
    return rgbState;
    }*/
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
  torch::Tensor stateInputs = torch::zeros({n,nTimesteps,3,size,size});
  torch::Tensor actionInputs = generateActions(n,nTimesteps);
  torch::Tensor stateLabels = torch::zeros({n,nTimesteps,3,size,size});
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
      
	  stateInputs[i][t] = toRGBTensor(torch::tensor(gw.getCurrentState().getStateVector()).unsqueeze(0))[0];
	  gw.setTakenAction(tensorToVector(ieActions[i][t]));
	  float r = gw.transition();
	  rewardLabels[i][t] = r;
	  if (r == WIN_REWARD)
	    {
	      hitsGoal = true;
	    }
	  stateLabels[i][t] = toRGBTensor(torch::tensor(gw.getCurrentState().getStateVector()).unsqueeze(0))[0];
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
