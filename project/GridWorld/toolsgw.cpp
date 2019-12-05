#include "toolsgw.h"
DEFINE_double(wp,0.1,"Percentage of forced win scenarios during the dataset generation"); 
DEFINE_bool(wn,false,"Adding white noise to the one-hot encoded action vectors");
DEFINE_double(sd,0.25,"Standard deviation");

ToolsGW::ToolsGW(){}

ToolsGW::ToolsGW(GridWorld gw): gw(gw){}

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


void ToolsGW::generateDataSet(string path, int nmaps, int n, int nTimesteps, float winProp)
{
  gw = GridWorld(path+"train/",nmaps);
  gw.generateVectorStates();
  
  //Initialising the tensors that will contain the training set

  int size = gw.getSize();
  torch::Tensor stateInputs = torch::zeros({4*n/5,nTimesteps,3,size,size});
  torch::Tensor actionInputs = torch::zeros({4*n/5,nTimesteps,4});
  torch::Tensor stateLabels = torch::zeros({4*n/5,nTimesteps,3,size,size});
  torch::Tensor rewardLabels = torch::zeros({4*n/5,nTimesteps});
   
  //Making the agent wander randomly for n episodes 

  int j=0;
  for (int i=0;i<n;i++)
    {
      
      //Displaying a progression bar in the terminal
      
      if (n > 100 && i%(5*n/100) == 0)
	{
	  cout << "Your agent is crashing into walls for science... " + to_string(i/(n/100)) + "%" << endl;
	}
      
      //Swapping to test set generation when training set generation is done
      
      if (i==4*n/5)
	{
	  gw = GridWorld(path+"test/",nmaps);
	  gw.generateVectorStates();
	  j = 0;
	  cout<< "Training set generation is complete! Now generating test set..."<<endl; 
	  torch::save(stateInputs,path+"stateInputsTrain.pt");
	  torch::save(actionInputs,path+"actionInputsTrain.pt");
	  torch::save(rewardLabels,path+"rewardLabelsTrain.pt");
	  torch::save(stateLabels,path+"stateLabelsTrain.pt");
	  stateInputs = torch::zeros({n/5,nTimesteps,3,size,size});
	  actionInputs = torch::zeros({n/5,nTimesteps,4});
	  stateLabels = torch::zeros({n/5,nTimesteps,3,size,size});
	  rewardLabels = torch::zeros({n/5,nTimesteps});
	}

      for (int t=0;t<nTimesteps;t++)
	{
      
	  //Building the dataset tensors
      
	  stateInputs[j][t] = toRGBTensor(torch::tensor(gw.getCurrentState().getStateVector()).unsqueeze(0))[0];
	  if (i>winProp*4*n/5 && i<n-winProp*n/5)
	    {
	      gw.setTakenAction(gw.randomAction()); //Not changing the action when in winning scenarios generation
	    }
	  actionInputs[j][t][(int)gw.getTakenAction()[0]]=1;
	  rewardLabels[j][t] = gw.transition();
	  stateLabels[j][t] = toRGBTensor(torch::tensor(gw.getCurrentState().getStateVector()).unsqueeze(0))[0];
	}            
      gw.reset();
      j++;
      
      //Adding win situations to the dataset as they occur rarely
	  
      if (i<winProp*4*n/5 || i>n-winProp*n/5)
	{
	  vector<bool> available = {gw.getObstacles()[gw.getGoalX()-1][gw.getGoalY()]==0, gw.getObstacles()[gw.getGoalX()][gw.getGoalY()+1]==0, gw.getObstacles()[gw.getGoalX()+1][gw.getGoalY()]==0, gw.getObstacles()[gw.getGoalX()][gw.getGoalY()-1]==0};	      	      
	  while (available == vector<bool>({false,false,false,false})) //Starting over if the map contains a goal surrounded by walls
	    {
	      cout<<"The goal is surrounded by walls !"<<endl;
	      gw.reset();
	      available = {gw.getObstacles()[gw.getGoalX()-1][gw.getGoalY()]==0, gw.getObstacles()[gw.getGoalX()][gw.getGoalY()+1]==0, gw.getObstacles()[gw.getGoalX()+1][gw.getGoalY()]==0, gw.getObstacles()[gw.getGoalX()][gw.getGoalY()-1]==0};
	    }
	  default_random_engine generator(random_device{}());
	  uniform_int_distribution<int> dist(0,3);
	  int picked = dist(generator);
	  while (!available[picked])
	    {
	      picked = dist(generator);
	    }
	  switch(picked)
	    {
	    case 0:
	      gw.setAgentX(gw.getGoalX()-1); gw.setAgentY(gw.getGoalY());
	      gw.setTakenAction({2});
	      break;
	    case 1:
	      gw.setAgentX(gw.getGoalX()); gw.setAgentY(gw.getGoalY()+1);
	      gw.setTakenAction({3});
	      break;
	    case 2:
	      gw.setAgentX(gw.getGoalX()+1); gw.setAgentY(gw.getGoalY());
	      gw.setTakenAction({0});
	      break;
	    case 3:
	      gw.setAgentX(gw.getGoalX()); gw.setAgentY(gw.getGoalY()-1);
	      gw.setTakenAction({1});
	      break;
	    }
	  gw.generateVectorStates();
	}      
    }
  
  //Saving the test set
  
  cout<< "Test set generation is complete!"<<endl; 
  
  torch::save(stateInputs,path+"stateInputsTest.pt");
  torch::save(actionInputs,path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(stateLabels,path+"stateLabelsTest.pt");
}

void ToolsGW::transitionAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  int n = testData.size(2);
  int m = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));
  vector<int> scores(2,0);
  vector<int> truth(2,0);

  for (int s=0;s<m;s++)
    {      
      for (int i=0;i<n;i++)
	{
	  for (int j=0;j<n;j++)
	    {
	      float pixelt = *testData[s][0][i][j].data<float>();
	      float pixell = *labels[s][0][i][j].data<float>();
		if (pixelt>0.7 && pixell<1.3)
		  {		  
		    pixelt=1;
		  }
		else
		  {
		    pixelt=0;
		  }
		truth[pixell]++;
		if (pixelt==pixell)
		  {
		    scores[pixell]++;
		  }
	    }
	}
    }
  cout<<"\n########## TRANSITION FUNCTION EVALUATION ##########\n"<<endl;
  vector<string> names = {"zeros", "ones"};
  for (int j=0;j<2;j++)
    {
      cout<<"Correctly classified " + names[j] + ": " + to_string(scores[j])+"/"+to_string(truth[j]) + " (" + to_string(100.*scores[j]/truth[j]) + "%)" << endl;
    }
  cout<<endl;
}

void ToolsGW::rewardAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  cout<<"e"<<endl;
  vector<int> rCounts(3,0);
  vector<int> scores(3,0);
  testData = testData.flatten().to(torch::Device(torch::kCPU));
  labels = labels.flatten();
  int m = testData.size(0);
  for (int s=0;s<m;s++)
    {
      float bug = EMPTY_SQUARE_REWARD;
      float rl = *labels[s].data<float>();
      if (rl==LOSE_REWARD)
	{
	  rCounts[0]++;
	}
      else if (rl == bug)
	{
	  rCounts[1]++;
	}
      else if (rl == WIN_REWARD)
	{
	  rCounts[2]++;
	}
      float precision = abs(*testData[s].data<float>()-rl);
      if (rl==LOSE_REWARD && precision<0.1)
	{
	  scores[0]++;
	}
      else if (rl == bug && precision<0.1 && *testData[s].data<float>()<0)
	{
	  scores[1]++;
	}
      else if (rl == WIN_REWARD && precision<0.1)
	{
	  scores[2]++;
	}
    }
  vector<string> text = {"LOSE REWARD","EMPTY SQUARE REWARD","WIN REWARD"};
  cout<<"\n########## REWARD FUNCTION EVALUATION ##########\n " << endl;
  for (int i=0;i<3;i++)
    {
      cout<<text[i]+": "+ to_string(scores[i]) + "/" + to_string(rCounts[i]) + " (" + to_string(100.*scores[i]/rCounts[i]) + "%)"<<endl;
    }
  cout<<endl;
}
