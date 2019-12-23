#include "toolsss.h"

ToolsSS::ToolsSS(){}

ToolsSS::ToolsSS(SpaceWorld sw): sw(sw){}

torch::Tensor ToolsSS::normalize(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,1);
  float vmax = 50;
  float size = 800;
  if (!reverse)    
    {
      y/=size;        
      y[2]=y[2]*size/vmax;
      y[3]=y[3]*size/vmax;
    }
  else
    {
      y*=size;        
      y[2]=y[2]/size*vmax;
      y[3]=y[3]/size*vmax;
    }
  y = y.transpose(0,1);		
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

  int i=0;
  int j=0;

  while(i<n)
    {      
      while(i<n && sw.epCount<EPISODE_LENGTH)
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
	      torch::save(stateInputs,path+"stateInputsTrain.pt");
	      torch::save(actionInputs,path+"actionInputsTrain.pt");
	      torch::save(rewardLabels,path+"rewardLabelsTrain.pt");
	      torch::save(stateLabels,path+"stateLabelsTrain.pt");
	      stateInputs = torch::zeros({n/5,nTimesteps,size});
	      actionInputs = torch::zeros({n/5,nTimesteps,6});
	      stateLabels = torch::zeros({n/5,nTimesteps,size});
	      rewardLabels = torch::zeros({n/5,nTimesteps});
	    }
	  	
	  for (int t=0;t<nTimesteps;t++)
	    {	    
	      //Building the dataset tensors
	      
	      stateInputs[j][t] = torch::tensor(sw.getCurrentState().getStateVector());	  
	      vector<float> a = sw.randomAction();	      
	      if (t!=0)
		{
		  vector<float> previousAction = sw.getTakenAction();
		  default_random_engine generator(random_device{}());
		  normal_distribution<float> dist(previousAction[1],SHIP_MAX_THRUST/10.);
		  float thrustPow = dist(generator);
		  if(thrustPow > SHIP_MAX_THRUST)
		    {
		      thrustPow = SHIP_MAX_THRUST;
		    }
		  if (thrustPow < 0)
		    {
		      thrustPow = 0;
		    }
		  a[1] = thrustPow;
		  dist = normal_distribution<float>(previousAction[2],M_PI/5);
		  float thrustOri = dist(generator);
		  if(thrustOri > 2*M_PI)
		    {
		      thrustOri -= 2*M_PI;
		    }
		  if (thrustOri < 0)
		    {
		      thrustOri += 2*M_PI;
		    }
		  a[2]=thrustOri;
		}
	      sw.setTakenAction(a);
	      actionInputs[j][t][(int)sw.getTakenAction()[0]]=1; //one-hot encoding
	      actionInputs[j][t][4]=sw.getTakenAction()[1];
	      actionInputs[j][t][5]=sw.getTakenAction()[2];
	      rewardLabels[j][t] = sw.transition();
	      stateLabels[j][t] = torch::tensor(sw.getCurrentState().getStateVector());	     
	    }
	  i++;
	  j++;
	}
      sw.reset();

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
  torch::save(stateInputs,path+"stateInputsTest.pt");
  torch::save(actionInputs,path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(stateLabels,path+"stateLabelsTest.pt");  
}

void ToolsSS::transitionAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  int s = testData.size(1);
  int n = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));
  vector<int> scores(4,0);
  //  cout<<torch::split(torch::split(testData,4,1)[0],20,0)[0]<<endl;
  //  cout<<torch::split(torch::split(labels,4,1)[0],20,0)[0]<<endl;
  for (int i=0;i<n;i++)
    {
      for (int j=0;j<2;j++)
	{
	  if (abs(*(testData[i][j]-labels[i][j]).data<float>())<5/800.)
	    {
	      scores[j]++;
	    }
	  if (abs(*(testData[i][j+2]-labels[i][j+2]).data<float>())<0.01)
	    {
	      scores[j+2]++;
	    }
	}
    }
  cout<<"\n########## TRANSITION FUNCTION EVALUATION (5 pixel tolerance) ##########\n"<<endl;
  vector<string> names = {"x", "y", "Vx", "Vy"};
  for (int j=0;j<4;j++)
    {
      cout<<"Correctly classified " + names[j] + ": " + to_string(scores[j])+"/"+to_string(n) + " (" + to_string(100.*scores[j]/n) + "%)" << endl;
    }
  cout<<endl;
  cout<< "POSITION MSE (TARGET: 0.00004): ";
  cout<<*torch::mse_loss(torch::split(testData,2,1)[0],torch::split(labels,2,1)[0]).data<float>()<<endl;
  cout<< "VELOCITY MSE (TARGET: 0.0001): ";
  cout<<*torch::mse_loss(torch::split(testData,2,1)[1],torch::split(labels,2,1)[1]).data<float>()<<endl;
  cout<<endl;  
  cout<<"################################################"<<endl;  
}

void ToolsSS::rewardAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  //  cout<<torch::split(testData,10,0)[1]<<endl;
  //  cout<<torch::split(labels,10,0)[1]<<endl;
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
  cout<< "REWARD MSE : ";
  cout<<*torch::mse_loss(testData,labels).data<float>()<<endl;
  cout<<endl;
  cout<<"################################################"<<endl;

}
