#include "toolsss.h"

ToolsSS::ToolsSS(){}

ToolsSS::ToolsSS(SpaceWorld sw): sw(sw){}

torch::Tensor ToolsSS::normalize(torch::Tensor x)
{
  int n = x.size(0), T = x.size(1), s = x.size(2);
  torch::Tensor y = x.clone();
  y/=sw.getSize();
  y = y.reshape({n*T,s}).transpose(0,1);
  torch::Tensor vmax = torch::max(y[2]*sw.getSize());
  y[2]=10*y[2]*sw.getSize()/vmax;
  y[3]=10*y[3]*sw.getSize()/vmax;
  y = y.transpose(0,1).reshape({n,T,s});
  return y;
}

void ToolsSS::generateDataSet(string path, int nmaps, int n, int nTimesteps, float winProp)
{
  sw = SpaceWorld(path+"train/",nmaps);

  //Initialising the tensors that will contain the training set

  int size = sw.getSvSize();
  torch::Tensor stateInputs = torch::zeros({4*n/5,nTimesteps,size});
  torch::Tensor actionInputs = torch::zeros({4*n/5,nTimesteps,6});
  torch::Tensor stateLabels = torch::zeros({4*n/5,nTimesteps,size});
  torch::Tensor rewardLabels = torch::zeros({4*n/5,nTimesteps});

  //Making the agent wander randomly for n episodes 
  
  int j=0;
  for (int i=0;i<n;i++)
    {
      
      //Displaying a progression bar in the terminal
      
      if (n > 100 && i%(5*n/100) == 0)
	{
	  cout << "Your agent is crashing into planets for science... " + to_string(i/(n/100)) + "%" << endl;
	}

      //Swapping to test set generation when training set generation is done
      
      if (i==4*n/5)
	{
	  sw = SpaceWorld(path+"test/",nmaps);
	  j = 0;
	  cout<< "Training set generation is complete! Now generating test set..."<<endl; 
	  torch::save(normalize(stateInputs),path+"stateInputsTrain.pt");
	  torch::save(actionInputs,path+"actionInputsTrain.pt");
	  torch::save(rewardLabels,path+"rewardLabelsTrain.pt");
	  torch::save(normalize(stateLabels),path+"stateLabelsTrain.pt");
	  stateInputs = torch::zeros({n/5,nTimesteps,size});
	  actionInputs = torch::zeros({n/5,nTimesteps,6});
	  stateLabels = torch::zeros({n/5,nTimesteps,size});
	  rewardLabels = torch::zeros({n/5,nTimesteps});
	}

      for (int t=0;t<nTimesteps;t++)
	{
      
	  //Building the dataset tensors
      
	  stateInputs[j][t] = torch::tensor(sw.getCurrentState().getStateVector());	  
	  sw.setTakenAction(sw.randomAction());
	  actionInputs[j][t][(int)sw.getTakenAction()[0]]=1; //one-hot encoding
	  actionInputs[j][t][4]=sw.getTakenAction()[1]/SHIP_MAX_THRUST;
	  actionInputs[j][t][5]=sw.getTakenAction()[2]/SHIP_MAX_THRUST;
	  rewardLabels[j][t] = sw.transition();
	  stateLabels[j][t] = torch::tensor(sw.getCurrentState().getStateVector());
	}
      sw.reset();
      j++;

      //Adding waypoint collision situations to the dataset as they occur more rarely
      
      if (i<winProp*4*n/5 || i>n-winProp*n/5)
	{
	  default_random_engine generator(random_device{}());
	  uniform_int_distribution<int> dist(0,sw.getWaypoints().size()-1);
	  int wpIdx = dist(generator);
	  sw.repositionShip(sw.getWaypoints()[wpIdx].getCentre());	        
	}
    }
      
  //Saving the test set

  cout<< "Test set generation is complete!"<<endl;
  torch::save(normalize(stateInputs),path+"stateInputsTest.pt");
  torch::save(actionInputs,path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(normalize(stateLabels),path+"stateLabelsTest.pt");  
}

void ToolsSS::transitionAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  int s = testData.size(1);
  int n = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));
  vector<int> scores(4,0);

  for (int i=0;i<n;i++)
    {
      for (int j=0;j<4;j++)
	{
	  if (abs(*(testData[i][j]-labels[i][j]).data<float>())<0.05*(*labels[i][j].data<float>()))
	    {
	      scores[j]++;
	    }
	}
    }
  cout<<"\n########## TRANSITION FUNCTION EVALUATION (5% tolerance) ##########\n"<<endl;
  vector<string> names = {"x", "y", "Vx", "Vy"};
  for (int j=0;j<4;j++)
    {
      cout<<"Correctly classified " + names[j] + ": " + to_string(scores[j])+"/"+to_string(n) + " (" + to_string(100.*scores[j]/n) + "%)" << endl;
    }
  cout<<endl;
}

void ToolsSS::rewardAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  vector<int> rCounts(4,0);
  vector<int> scores(4,0);
  testData = testData.flatten().to(torch::Device(torch::kCPU));
  labels = labels.flatten();
  int m = testData.size(0);
  for (int s=0;s<m;s++)
    {
      float rl = *labels[s].data<float>();
      if (rl==CRASH_REWARD)
	{
	  rCounts[0]++;
	}
      else if (abs(rl-SIGNAL_OFF_WAYPOINT_REWARD)<0.001) //had to do this because of -0.1 float approximation
	{
	  rCounts[1]++;
	}
      else if (rl == RIGHT_SIGNAL_ON_WAYPOINT_REWARD)
	{
	  rCounts[2]++;
	}
      else if (rl==0)
	{
	  rCounts[3]++;
	}
      float precision = abs(*testData[s].data<float>()-rl);
      if (rl==CRASH_REWARD && precision<0.1)
	{
	  scores[0]++;
	}
      else if (abs(rl-SIGNAL_OFF_WAYPOINT_REWARD)<0.001 && precision<0.05)
	{
	  scores[1]++;
	}
      else if (rl == RIGHT_SIGNAL_ON_WAYPOINT_REWARD && precision<0.1)
	{
	  scores[2]++;
	}
      else if (rl == 0 && precision<0.05)
	{
	  scores[3]++;
	}
    }
  vector<string> text = {"CRASH OR WRONG SIGNAL ON WAYPOINT","SIGNAL OFF WAYPOINT","RIGHT SIGNAL ON WAYPOINT","DID NOTHING"};
  cout<<"\n########## REWARD FUNCTION EVALUATION ##########\n " << endl;
  for (int i=0;i<4;i++)
    {
      cout<<text[i]+": "+ to_string(scores[i]) + "/" + to_string(rCounts[i]) + " (" + to_string(100.*scores[i]/rCounts[i]) + "%)"<<endl;
    }
  cout<<endl;
}
