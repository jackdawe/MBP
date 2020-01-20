#include "toolsss.h"

ToolsSS::ToolsSS():
  tScores(vector<int>(4,0)), rScores(vector<int>(4,0)), rCounts(vector<int>(4,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

ToolsSS::ToolsSS(SpaceWorld sw):
  tScores(vector<int>(4,0)), rScores(vector<int>(4,0)), rCounts(vector<int>(4,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

torch::Tensor ToolsSS::normalizeStates(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,1);
  float vmax = 20;
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

torch::Tensor ToolsSS::normalizeDeltas(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,1);
  float vmax = 10;
  float size = 50;
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

torch::Tensor ToolsSS::normalizeActions(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,-1);
  if (reverse)
    {
      y[4]*=SHIP_MAX_THRUST;
      y[5]*=SHIP_MAX_THRUST;
    }
  else
    {
      y[4]/=SHIP_MAX_THRUST;
      y[5]/=SHIP_MAX_THRUST;     
    }
  return y.transpose(0,-1);
}

torch::Tensor ToolsSS::deltaToState(torch::Tensor stateBatch, torch::Tensor deltas)
{
  torch::Tensor pos = stateBatch.slice(1,0,2,1);
  torch::Tensor velo = stateBatch.slice(1,2,4,1);  
  torch::Tensor constPart = stateBatch.slice(1,4,stateBatch.size(1),1);  
  torch::Tensor newStates = torch::cat({torch::remainder(pos+deltas.slice(1,0,2,1),800),velo+deltas.slice(1,2,4,1),constPart},1);
  return newStates;
}

torch::Tensor ToolsSS::moduloMSE(torch::Tensor x, torch::Tensor target, bool normalized)
{
  int bound = 1*normalized+800*(1-normalized);
  torch::Tensor compare = torch::cat({x.unsqueeze(0),target.unsqueeze(0)},0);
  //  cout<<compare.transpose(0,1)[0]<<endl;
  torch::Tensor mini = get<0>(torch::min(compare,0));
  torch::Tensor maxi = get<0>(torch::max(compare,0));
  torch::Tensor a = (mini+bound-maxi).unsqueeze(0);
  compare = torch::cat({a,bound-a});
  //  cout<<get<0>(torch::min(compare,0))[0]<<endl;
  return (get<0>(torch::min(compare,0)).pow(2).mean());
}

void ToolsSS::generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp, float edgeSpawnProp)
{
  cout<<"Generating a dataset for the Starship task containing " + to_string(n) + " samples of " + to_string(nTimesteps) + " time steps. An episode ends after "+to_string(EPISODE_LENGTH)+" time steps."<<endl;
  cout<<"The training set contains " +to_string((int)(100*trainSetProp))+"% of the dataset and the test set the remaining samples."<<endl;
  cout<<"To help with the training, the agent is forced to spawn on a waypoint in " + to_string((int)(100*winProp)) + "% of the episodes."<<endl;
  cout<<"To help with the training, the agent is forced to spawn near the edge of the map in " + to_string((int)(100*edgeSpawnProp)) + "% of the episodes."<<endl;  

  sw = SpaceWorld(path+"train/",nmaps);
  int nTr=(int)(trainSetProp*n), nTe = n-nTr;
  
  //Initialising the tensors that will contain the training set

  int size = sw.getSvSize();
  torch::Tensor stateInputs = torch::zeros({nTr,nTimesteps,size});
  torch::Tensor actionInputs = torch::zeros({nTr,nTimesteps,6});
  torch::Tensor stateLabels = torch::zeros({nTr,nTimesteps,4});
  torch::Tensor rewardLabels = torch::zeros({nTr,nTimesteps});

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
	      cout << "Your agent is crashing into planets for science... " + to_string((int)(i/(n/100.))) + "%" << endl;
	    }	  
	  
	  //Swapping to test set generation when training set generation is done
	  
	  if (i==nTr)
	    {
	      sw = SpaceWorld(path+"test/",nmaps);
	      j = 0;
	      cout<< "Training set generation is complete! Now generating test set..."<<endl; 	      
	      torch::save(stateInputs,path+"stateInputsTrain.pt");
	      torch::save(normalizeActions(actionInputs),path+"actionInputsTrain.pt");
	      torch::save(rewardLabels,path+"rewardLabelsTrain.pt");
	      torch::save(stateLabels,path+"stateLabelsTrain.pt");
	      stateInputs = torch::zeros({nTe,nTimesteps,size});
	      actionInputs = torch::zeros({nTe,nTimesteps,6});
	      stateLabels = torch::zeros({nTe,nTimesteps,4});
	      rewardLabels = torch::zeros({nTe,nTimesteps});	      
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
		  a[1] = dist(generator);
		  a[2] = dist(generator);		  
		}
	      sw.setTakenAction(a);
	      actionInputs[j][t][(int)sw.getTakenAction()[0]]=1; //one-hot encoding
	      actionInputs[j][t][4]=sw.getTakenAction()[1];
	      actionInputs[j][t][5]=sw.getTakenAction()[2];
	      rewardLabels[j][t] = sw.transition();

	      //torch::Tensor nextState = torch::zeros({4});
	      /*	      nextState[0] = sw.getPreviousState().getStateVector()[2]*STEP_SIZE;
	      nextState[1] = sw.getPreviousState().getStateVector()[3]*STEP_SIZE;
	      nextState[2] = sw.getShip().getA().x*STEP_SIZE;
	      nextState[3] = sw.getShip().getA().y*STEP_SIZE;	      */
	      vector<float> nextStateVec  = sw.getCurrentState().getStateVector();
	      stateLabels[j][t] = torch::tensor(vector<float>(nextStateVec.begin(),nextStateVec.begin()+4));
	    }
	  i++;
	  j++;
	}
      sw.reset();

      //Adding waypoint collision situations to the dataset as they occur more rarely
      
      if (i<=winProp*nTr || i>=n-winProp*nTe)
	{
	  default_random_engine generator(random_device{}());
	  uniform_int_distribution<int> dist(0,sw.getWaypoints().size()-1);
	  int wpIdx = dist(generator);
	  sw.repositionShip(sw.getWaypoints()[wpIdx].getCentre());	        
	  sw.generateVectorStates();
	}

      if ((i>winProp*nTr && i<=(edgeSpawnProp+winProp)*nTr) || (i<n-winProp*nTe && i>=n-(edgeSpawnProp+winProp)*nTe))
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
  torch::save(normalizeActions(actionInputs),path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(stateLabels,path+"stateLabelsTest.pt");    
}

void ToolsSS::transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit)
{
  int s = testData.size(1);
  int n = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));
  //  cout<<torch::cat({testData.slice(1,0,2,1), labels.slice(1,0,2,1)},1)<<endl;
  /*  for (int i=0;i<n;i++)
    {
      torch::Tensor a = moduloMSE(testData.slice(1,0,2,1)[i],labels.slice(1,0,2,1)[i],false);
      if (*a.data<float>()>100000)
	{
	  cout<<torch::cat({testData.slice(1,0,2,1), labels.slice(1,0,2,1)},1)[i].unsqueeze(0)<<endl;   
	}
    }
  */
  pMSE+=moduloMSE(testData.slice(1,0,2,1),labels.slice(1,0,2,1),false)/nSplit;
  vMSE+=torch::mse_loss(testData.slice(1,2,4,1),labels.slice(1,2,4,1))/nSplit;
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
  cout<< "POSITION DELTA AVERAGE ERROR: ";
  cout<<pow(*pMSE.data<float>(),0.5)<<endl;
  cout<< "VELOCITY DELTA AVERAGE ERROR: ";
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
  /*  for (int s=0;s<m;s++)
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
