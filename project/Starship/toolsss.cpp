#include "toolsss.h"

ToolsSS::ToolsSS():
  tScores(vector<int>(4,0)), rScores(vector<int>(5,0)), rCounts(vector<int>(5,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

ToolsSS::ToolsSS(SpaceWorld sw):
  sw(sw), tScores(vector<int>(4,0)), rScores(vector<int>(5,0)), rCounts(vector<int>(5,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

vector<float> ToolsSS::tensorToVector(torch::Tensor stateVector)
{
  vector<float> vec;
  for (int i=0;i<stateVector.size(0);i++)
    {
      vec.push_back(*stateVector[i].data<float>());
    }
  return vec;
}

torch::Tensor ToolsSS::normalizeStates(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,-1);
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
  y = y.transpose(0,-1);		
  return y;
}

torch::Tensor ToolsSS::normalizeDeltas(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,-1);
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
  y = y.transpose(0,-1);		
  return y;
}

torch::Tensor ToolsSS::normalizeActions(torch::Tensor x, bool reverse)
{
  torch::Tensor y = x.clone();
  y = y.transpose(0,-1);
  if (reverse)
    {
      y[4]=y[4]*2*SHIP_MAX_THRUST-0.5;
      y[5]=y[5]*2*SHIP_MAX_THRUST-0.5;
    }
  else
    {
      y[4]=y[4]/(2*SHIP_MAX_THRUST)+0.5;
      y[5]=y[5]/(2*SHIP_MAX_THRUST)+0.5;
    }
  return y.transpose(0,-1);
}

torch::Tensor ToolsSS::deltaToState(torch::Tensor stateBatch, torch::Tensor deltas)
{
  torch::Tensor pos = stateBatch.slice(1,0,2,1);
  torch::Tensor velo = stateBatch.slice(1,2,4,1);  
  torch::Tensor constPart = stateBatch.slice(1,4,stateBatch.size(1),1);  
  torch::Tensor newStates = torch::cat({torch::remainder(pos+deltas.slice(1,0,2,1),800),torch::clamp(velo+deltas.slice(1,2,4,1),-20,20),constPart},1);
  return newStates;
}

torch::Tensor ToolsSS::moduloMSE(torch::Tensor target, torch::Tensor label, bool normalized)
{
  int bound = 1*normalized+800*(1-normalized);
  torch::Tensor compare = torch::cat({target.unsqueeze(0),label.unsqueeze(0)},0);
  torch::Tensor mini = get<0>(torch::min(compare,0));
  torch::Tensor maxi = get<0>(torch::max(compare,0));
  torch::Tensor a = (mini+bound-maxi).unsqueeze(0);
  compare = torch::cat({a,bound-a});
  return (get<0>(torch::min(compare,0)).pow(2).mean());
}

torch::Tensor ToolsSS::penalityMSE(torch::Tensor target, torch::Tensor label, float valToPenalize, float weight)
{
  torch::Tensor selected = torch::eq(label,torch::full(target.sizes(),valToPenalize).to(label.device()));  
  int nValToPenalize = *selected.sum().to(torch::Device(torch::kCPU)).data<long>(); 
  if (nValToPenalize==0)
    {
      return torch::zeros({1}).to(target.device());
    }
  else
    {
      return weight*((label-target)*selected).pow(2).sum()/nValToPenalize;
    }
}

torch::Tensor ToolsSS::generateActions(int n, int nTimesteps, int distribution, float alpha, float std)
{
  torch::Tensor thrust;
  switch(distribution)
    {
    case 0: //Uniform cartesian coordinate distribution
      {
	thrust = torch::rand({n,nTimesteps,2})*2*SHIP_MAX_THRUST-SHIP_MAX_THRUST;
      }
      break;
    case 1: //Uniform polar coordinate distribution
      {
	torch::Tensor r = torch::rand({n,nTimesteps,1})*SHIP_MAX_THRUST;
	torch::Tensor theta = torch::rand({n,nTimesteps,1})*2*M_PI;
	thrust = torch::cat({r*torch::cos(theta),r*torch::sin(theta)},-1);
      }
      break;
    case 2: //Gaussian cartesian coordinate distribution
      {
	torch::Tensor centre = torch::rand({n,2})*2*SHIP_MAX_THRUST-SHIP_MAX_THRUST;
	thrust = torch::zeros({n,nTimesteps,2});
	for (int i=0;i<n;i++)
	  {
	    thrust[i] = torch::cat({torch::zeros({nTimesteps,1}).normal_(*centre[i][0].data<float>(),std),torch::zeros({nTimesteps,1}).normal_(*centre[i][1].data<float>(),std)},-1);
	  }
      }
      break;
    case 3: //Gaussian polar coordinate distribution
      {
	torch::Tensor centreR = torch::rand({n});
	torch::Tensor centreTheta = torch::rand({n});            
	thrust = torch::zeros({n,nTimesteps,2});
	for (int i=0;i<n;i++)
	  {
	    thrust[i] = torch::cat({torch::zeros({nTimesteps,1}).normal_(*centreR[i].data<float>(),std),torch::zeros({nTimesteps,1}).normal_(*centreTheta[i].data<float>(),std)},-1);
	  }
	torch::Tensor r = thrust.slice(-1,0,1,1)*SHIP_MAX_THRUST;
	torch::Tensor theta = thrust.slice(-1,1,2,1)*2*M_PI;
	thrust = torch::cat({r*torch::cos(theta),r*torch::sin(theta)},-1);
      }
      break;      
    default:      
      thrust = torch::ones({n,nTimesteps,2});
    }
  torch::Tensor signal = torch::zeros({4,nTimesteps,n});
  signal = signal.scatter_(0,torch::randint(0,4,{1,nTimesteps,n}).to(torch::kLong),torch::ones_like(signal)).transpose(0,2);
  return torch::cat({signal,alpha*thrust},-1);
}


void ToolsSS::generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp, int aDist, float alpha, float std)
{

  if (EPISODE_LENGTH % nTimesteps!=0)
    {
      cout<<"Impossible to make samples of equal length. Please chose nTimesteps such as EPISODE_LENGTH%nTimesteps == 0"<<endl;
    }
  else
    {
      cout<<"Generating a dataset for the Starship task containing " + to_string(n) + " samples of " + to_string(nTimesteps) + " time steps. An episode ends after "+to_string(EPISODE_LENGTH)+" time steps."<<endl;
      cout<<"The training set contains " +to_string((int)(100*trainSetProp))+"% of the dataset and the test set the remaining samples."<<endl;
      cout<<"To help with the training, only samples containing at least one waypoint connexion are kept in " + to_string((int)(100*winProp)) + "% of the episodes."<<endl;
      
      int nSplits = EPISODE_LENGTH/nTimesteps;
      
      sw = SpaceWorld(path+"train/",nmaps);
      int nTr=(int)(trainSetProp*n*nSplits), nTe = nSplits*n-nTr;
      
      //Initialising the tensors that will contain the training set
      
      int size = sw.getSvSize();
      torch::Tensor stateInputs = torch::zeros({n*nSplits,size});
      torch::Tensor actionInputs = generateActions(n*nSplits,nTimesteps, aDist,alpha, std);
      torch::Tensor stateLabels = torch::zeros({n*nSplits,nTimesteps,4});
      torch::Tensor rewardLabels = torch::zeros({n*nSplits,nTimesteps});
      
      //Generating another action tensor where signal is encoded as an int
      
      torch::Tensor signalInt = torch::argmax(actionInputs.slice(-1,0,4,1),-1).to(torch::kFloat32);
      torch::Tensor ieActions = torch::cat({signalInt.unsqueeze(-1),actionInputs.slice(-1,4,6,1)},-1);
      
      //Making the agent wander randomly for n episodes 
      
      int i=0;
      bool dispPerc = true;
      while(i<n)
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
	      sw = SpaceWorld(path+"test/",nmaps);
	    }            
	  torch::Tensor si = torch::zeros(0);      
	  torch::Tensor sl = torch::zeros({EPISODE_LENGTH,4});
	  torch::Tensor rl = torch::zeros({EPISODE_LENGTH});      
	  bool hitsWayp=false;
	  for (int t=0;t<EPISODE_LENGTH;t++)	  	  
	    {	    
	      //Building the dataset tensors
	      
	      if(t%nTimesteps == 0)
		{
		  si = torch::cat({si,torch::tensor(sw.getCurrentState().getStateVector()).unsqueeze(0)},0);
		}
	      float r = sw.transition(tensorToVector(ieActions[nSplits*i+t/nTimesteps][t%nTimesteps]));
	      rl[t] = r;	      
	      if (r == RIGHT_SIGNAL_ON_WAYPOINT_REWARD || r == WRONG_SIGNAL_ON_WAYPOINT_REWARD)
		{
		  hitsWayp = true;
		}
	      vector<float> nextStateVec  = sw.getCurrentState().getStateVector();
	      sl[t] = torch::tensor(vector<float>(nextStateVec.begin(),nextStateVec.begin()+4));
	    }
	  
	  if ((i>=winProp*nTr/nSplits && i<=n-winProp*nTe/nSplits) || hitsWayp)
	    {	  		
	      vector<torch::Tensor> slSplit = torch::split(sl,nTimesteps,0);   
	      vector<torch::Tensor> rlSplit = torch::split(rl,nTimesteps,0);
	      for (int j=0;j<nSplits;j++)
		{
		  stateInputs[i*nSplits+j]=si[j];
		  stateLabels[i*nSplits+j]=slSplit[j];
		  rewardLabels[i*nSplits+j]=rlSplit[j];	  
		}
	      i++;
	      dispPerc = true;
	      hitsWayp = false;
	    }
	  sw.reset();
	}
      //Saving the datatest set
      
      cout<< "Data set generation is complete!"<<endl;
      torch::save(stateInputs.slice(0,0,nTr,1),path+"stateInputsTrain.pt");
      torch::save(normalizeActions(actionInputs).slice(0,0,nTr,1),path+"actionInputsTrain.pt");
      torch::save(rewardLabels.slice(0,0,nTr,1),path+"rewardLabelsTrain.pt");
      torch::save(stateLabels.slice(0,0,nTr,1),path+"stateLabelsTrain.pt");    
      torch::save(stateInputs.slice(0,nTr,nTr+nTe,1),path+"stateInputsTest.pt");
      torch::save(normalizeActions(actionInputs).slice(0,nTr,nTr+nTe,1),path+"actionInputsTest.pt");
      torch::save(rewardLabels.slice(0,nTr,nTr+nTe,1),path+"rewardLabelsTest.pt");
      torch::save(stateLabels.slice(0,nTr,nTr+nTe,1),path+"stateLabelsTest.pt");  
    }
}
  
float ToolsSS::comparePosMSE(torch::Tensor initState, int nWaypoints, torch::Tensor actionSequence, torch::Tensor estimate)
{
  sw = SpaceWorld(tensorToVector(initState),nWaypoints);
  int T = actionSequence.size(0);
  actionSequence = torch::cat({torch::zeros({T,1}),actionSequence},-1);
  torch::Tensor labels = torch::zeros({T,initState.size(0)});
  for (int t=0;t<T;t++)
    {
      sw.transition(tensorToVector(actionSequence[t]));
      labels[t] = torch::tensor(sw.getCurrentState().getStateVector());
    }
  return *moduloMSE(estimate.slice(-1,0,2,1),labels.slice(-1,0,2,1),false).data<float>();
}

void ToolsSS::generateSeed(int nTimesteps, int nRollouts, string filename)
{
  torch::Tensor actions = torch::cat({torch::zeros({nTimesteps,nRollouts,4}).normal_(0,1),torch::rand({nTimesteps,nRollouts,2})},2); 
  cout<<actions<<endl;
  torch::save(actions,filename+".pt");
}

void ToolsSS::transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit, bool disp)
{
  pMSE+=moduloMSE(testData.slice(-1,0,2,1),labels.slice(-1,0,2,1),false)/nSplit;
  vMSE+=torch::mse_loss(testData.slice(-1,2,4,1),labels.slice(-1,2,4,1))/nSplit;  
  testData = testData.to(torch::Device(torch::kCPU));
  labels = labels.to(torch::Device(torch::kCPU));  
  if (disp)
    {
      vector<float> thresholds = {5,5,0.2,0.2};
      for (int i=0;i<4;i++)
	{
	  torch::Tensor precision = torch::abs(testData.slice(-1,i,i+1,1)-labels.slice(-1,i,i+1,1));
	  tScores[i] += *(torch::lt(precision,thresholds[i]).sum()).data<long>();
	}
    }
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
  cout<< "POSITION AVERAGE ERROR: ";
  cout<<pow(*pMSE.data<float>(),0.5)<<endl;
  cout<< "VELOCITY DELTA ERROR: ";
  cout<<pow(*vMSE.data<float>(),0.5)<<endl;
  cout<<endl;  
  cout<<"################################################"<<endl;  
}


void ToolsSS::rewardAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit, bool disp)
{
  cout<<penalityMSE(testData,labels,RIGHT_SIGNAL_ON_WAYPOINT_REWARD,1).pow(0.5)<<endl;
  rMSE+=torch::mse_loss(testData,labels)/nSplit;
  testData = testData.to(torch::Device(torch::kCPU));
  labels = labels.to(torch::Device(torch::kCPU));  
  if (disp)
    {
      vector<float> rewards = {CRASH_REWARD,SIGNAL_OFF_WAYPOINT_REWARD,RIGHT_SIGNAL_ON_WAYPOINT_REWARD,0,WRONG_SIGNAL_ON_WAYPOINT_REWARD};
      torch::Tensor precision = torch::abs(testData-labels);
      vector<float> thresholds = {0.2,0.1,0.2,0.1,0.2};
      for (int i=0;i<5;i++)
	{
	  torch::Tensor forCounting = torch::eq(labels,torch::full(labels.sizes(),rewards[i]));
	  rCounts[i] += *forCounting.sum().data<long>();
	  rScores[i] += *(torch::lt(precision,torch::full(labels.sizes(),thresholds[i])*forCounting)).sum().data<long>();
	}
    }
}

void ToolsSS::displayRAccuracy()
{
  vector<string> text = {"CRASH","SIGNAL OFF WAYPOINT","RIGHT SIGNAL ON WAYPOINT","DID NOTHING","WRONG SIGNAL ON WAYPOINT"};
  cout<<"\n########## REWARD FUNCTION EVALUATION ##########\n " << endl;
  for (int i=0;i<5;i++)
    {
      cout<<text[i]+": "+ to_string(rScores[i]) + "/" + to_string(rCounts[i]) + " (" + to_string(100.*rScores[i]/rCounts[i]) + "%)"<<endl;
    }
  cout<<endl;
  cout<< "REWARD AVERAGE ERROR : ";
  cout<<pow(*rMSE.data<float>(),0.5)<<endl;
  cout<<endl;
  cout<<"################################################"<<endl;

}
