#include "toolsss.h"

ToolsSS::ToolsSS():
  tScores(vector<int>(4,0)), rScores(vector<int>(4,0)), rCounts(vector<int>(4,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

ToolsSS::ToolsSS(SpaceWorld sw):
  tScores(vector<int>(4,0)), rScores(vector<int>(4,0)), rCounts(vector<int>(4,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

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

void ToolsSS::generateDataSet(string path, int nmaps, int n, int nTimesteps, float winProp, float edgeSpawnProp)
{
  cout<<"Generating a dataset for the Starship task containing " + to_string(n) + " samples of " + to_string(nTimesteps) + " time steps. An episode ends after "+to_string(EPISODE_LENGTH)+" time steps."<<endl;
  cout<<"The training set contains 80% of the dataset and the test set the remaining samples."<<endl;
  cout<<"To help with the training, the agent is forced to spawn on a waypoint in " + to_string((int)(100*winProp)) + "% of the episodes."<<endl;
  cout<<"To help with the training, the agent is forced to spawn near the edge of the map in " + to_string((int)(100*edgeSpawnProp)) + "% of the episodes."<<endl;  

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
		  a[1] = thrustPow;
		  dist = normal_distribution<float>(previousAction[2],M_PI/5);
		  float thrustOri = dist(generator);
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
      
      if (i<=winProp*4*n/5 || i>=n-winProp*n/5)
	{
	  default_random_engine generator(random_device{}());
	  uniform_int_distribution<int> dist(0,sw.getWaypoints().size()-1);
	  int wpIdx = dist(generator);
	  sw.repositionShip(sw.getWaypoints()[wpIdx].getCentre());	        
	  sw.generateVectorStates();
	}

      if ((i>winProp*4*n/5 && i<=(edgeSpawnProp+winProp)*4*n/5) || (i<n-winProp*n/5 && i>=n-(edgeSpawnProp+winProp)*n/5))
	{
	  default_random_engine generator(random_device{}());
	  uniform_int_distribution<int> dist(0,1);
	  bool isX = dist(generator);
	  bool nearZero = dist(generator);
	  dist = uniform_int_distribution<int>(0,10);
	  int newCoordinate = dist(generator);
	  if (isX)
	    {
	      if (nearZero)
		{
		  sw.repositionShip(Vect2d(newCoordinate,sw.getShip().getP().y));
		}
	      else
		{
		  sw.repositionShip(Vect2d(800-newCoordinate,sw.getShip().getP().y));
		}	      
	    }
	  else
	    {
	      if (nearZero)
		{
		  sw.repositionShip(Vect2d(sw.getShip().getP().x,newCoordinate));
		}
	      else
		{
		  sw.repositionShip(Vect2d(sw.getShip().getP().x,800-newCoordinate));
		}	      	      
	    }
	  sw.generateVectorStates();
	}      
    }
      
  //Saving the test set

  cout<< "Test set generation is complete!"<<endl;
  torch::save(stateInputs,path+"stateInputsTest.pt");
  torch::save(actionInputs,path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(stateLabels,path+"stateLabelsTest.pt");  
}

void ToolsSS::transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit)
{
  int s = testData.size(1);
  int n = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));

  pMSE+=torch::mse_loss(torch::split(testData,2,1)[0],torch::split(labels,2,1)[0])/nSplit;
  vMSE+=torch::mse_loss(torch::split(testData,2,1)[1],torch::split(labels,2,1)[1])/nSplit;

  //  cout<<torch::split(torch::split(testData,4,1)[0],20,0)[0]<<endl;
  //  cout<<torch::split(torch::split(labels,4,1)[0],20,0)[0]<<endl;


  //  torch::Tensor td = torch::split(testData.slice(0,190000,200000,1),1,1)[0];
  //  torch::Tensor ll = torch::split(labels.slice(0,190000,200000,1),1,1)[0];
  //  torch::Tensor aa = torch::arange(190000,200000,1)/1000.unsqueeze(1);
  //  cout<<torch::cat({td,ll},1)<<endl;
  /*
  torch::Tensor diff = torch::abs(testData-labels);
  int ft=0;
  for (int i=10;i<n-10;i++)
    {
      if (*diff[i][1].data<float>()>150)
	{
	  ft++;
	  torch::Tensor pp = torch::split(testData.slice(0,i-2,i+2,1),4,1)[0];
	  torch::Tensor qq = torch::split(labels.slice(0,i-2,i+2,1),4,1)[0];	  
	  torch::Tensor rr = (torch::arange(i-10,i+10,1)).unsqueeze(1);
	  cout<<torch::cat({pp,qq},1)<<endl;
	}      
    }
  cout<<ft<<endl;
  */
  /*
  torch::Tensor distance = torch::abs(testData-labels);   
  for (int i=0;i<n;i++)
    {
      for (int j=0;j<2;j++)
	{
	  if (*distance[i][j].data<float>()<5)
	    {
	      tScores[j]++;
	    }
	  if (*distance[i][j+2].data<float>()<0.2)
	    {
	      tScores[j+2]++;
	    }
	}
    }
  */
}

void ToolsSS::displayTAccuracy(int dataSetSize)
{
  cout<<"\n########## TRANSITION FUNCTION EVALUATION (5 pixel tolerance) ##########\n"<<endl;
  vector<string> names = {"x", "y", "Vx", "Vy"};
  for (int j=0;j<4;j++)
    {
      cout<<"Correctly classified " + names[j] + ": " + to_string(tScores[j])+"/"+to_string(dataSetSize) + " (" + to_string(100.*tScores[j]/dataSetSize) + "%)" << endl;
    }
  cout<<endl;
  cout<< "POSITION AVERAGE ERROR (TARGET: 5 pixels): ";
  cout<<pow(*pMSE.data<float>(),0.5)<<endl;
  cout<< "VELOCITY MSE (TARGET: 0.2 pixels per step): ";
  cout<<pow(*vMSE.data<float>(),0.5)<<endl;
  cout<<endl;  
  cout<<"################################################"<<endl;  
}


void ToolsSS::rewardAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit)
{
  testData = testData.flatten().to(torch::Device(torch::kCPU));
  labels = labels.flatten();
  int m = testData.size(0);
  rMSE+=torch::mse_loss(testData,labels)/nSplit;
  /*
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
	  rScores[0]++;
	}
      else if (abs(rl-SIGNAL_OFF_WAYPOINT_REWARD)<0.001 && precision<0.05)
	{
	  rScores[1]++;
	}
      else if (rl == RIGHT_SIGNAL_ON_WAYPOINT_REWARD && precision<0.1)
	{
	  rScores[2]++;
	}
      else if (rl == 0 && precision<0.05)
	{
	  rScores[3]++;
	}
    }
  */
}

void ToolsSS::displayRAccuracy()
{
  vector<string> text = {"CRASH OR WRONG SIGNAL ON WAYPOINT","SIGNAL OFF WAYPOINT","RIGHT SIGNAL ON WAYPOINT","DID NOTHING"};
  cout<<"\n########## REWARD FUNCTION EVALUATION ##########\n " << endl;
  for (int i=0;i<4;i++)
    {
      cout<<text[i]+": "+ to_string(rScores[i]) + "/" + to_string(rCounts[i]) + " (" + to_string(100.*rScores[i]/rCounts[i]) + "%)"<<endl;
    }
  cout<<endl;
  cout<< "REWARD AVERAGE ERROR : ";
  cout<<pow(*rMSE.data<float>(),1)<<endl;
  cout<<endl;
  cout<<"################################################"<<endl;

}
