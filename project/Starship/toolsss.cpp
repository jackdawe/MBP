#include "toolsss.h"

ToolsSS::ToolsSS():
  tScores(vector<int>(4,0)), rScores(vector<int>(5,0)), rCounts(vector<int>(5,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
{}

ToolsSS::ToolsSS(SpaceWorld sw):
  tScores(vector<int>(4,0)), rScores(vector<int>(5,0)), rCounts(vector<int>(5,0)), pMSE(torch::zeros({1})), vMSE(torch::zeros({1})), rMSE(torch::zeros({1}))
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
  torch::Tensor newStates = torch::cat({torch::remainder(pos+deltas.slice(1,0,2,1),800),velo+deltas.slice(1,2,4,1),constPart},1);
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
      return torch::zeros({1});
    }
  else
    {
      return weight*((label-target)*selected).pow(2).sum()/nValToPenalize;
    }
}



void ToolsSS::generateDataSet(string path, int nmaps, int n, int nTimesteps, float trainSetProp, float winProp)
{
  cout<<"Generating a dataset for the Starship task containing " + to_string(n) + " samples of " + to_string(nTimesteps) + " time steps. An episode ends after "+to_string(EPISODE_LENGTH)+" time steps."<<endl;
  cout<<"The training set contains " +to_string((int)(100*trainSetProp))+"% of the dataset and the test set the remaining samples."<<endl;
  cout<<"To help with the training, the agent is forced to spawn on a waypoint in " + to_string((int)(100*winProp)) + "% of the episodes."<<endl;

  if (EPISODE_LENGTH % nTimesteps!=0)
    {
      cout<<"Warning: impossible to make samples of equal length."<<endl;
    }
  int nSplits = EPISODE_LENGTH/nTimesteps;
  
  sw = SpaceWorld(path+"train/",nmaps);
  int nTr=(int)(trainSetProp*n*nSplits), nTe = nSplits*n-nTr;
  
  //Initialising the tensors that will contain the training set

  int size = sw.getSvSize();
  torch::Tensor stateInputs = torch::zeros({n*nSplits,nTimesteps,size});
  torch::Tensor actionInputs = torch::zeros({n*nSplits,nTimesteps,6});
  torch::Tensor stateLabels = torch::zeros({n*nSplits,nTimesteps,4});
  torch::Tensor rewardLabels = torch::zeros({n*nSplits,nTimesteps});

  //Making the agent wander randomly for n episodes 

  //  bool dispPerc = true;
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
      torch::Tensor si = torch::zeros({EPISODE_LENGTH,size});
      torch::Tensor ai = torch::zeros({EPISODE_LENGTH,6});
      torch::Tensor sl = torch::zeros({EPISODE_LENGTH,4});
      torch::Tensor rl = torch::zeros({EPISODE_LENGTH});      
      bool hitsWayp=false;
      for (int t=0;t<EPISODE_LENGTH;t++)	  	  
	{	    
	  //Building the dataset tensors
	  
	  si[t] = torch::tensor(sw.getCurrentState().getStateVector());	  
	  vector<float> a = sw.randomAction();	      	  	  
	  sw.setTakenAction(a);
	  ai[t][(int)sw.getTakenAction()[0]]=1; //one-hot encoding
	  ai[t][4]=sw.getTakenAction()[1];
	  ai[t][5]=sw.getTakenAction()[2];
	  float r = sw.transition();
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
	  vector<torch::Tensor> siSplit = torch::split(si,nTimesteps,0);
	  vector<torch::Tensor> aiSplit = torch::split(ai,nTimesteps,0);
	  vector<torch::Tensor> slSplit = torch::split(sl,nTimesteps,0);   
	  vector<torch::Tensor> rlSplit = torch::split(rl,nTimesteps,0);
	  for (int j=0;j<nSplits;j++)
	    {
	      stateInputs[i*nSplits+j]=siSplit[j];
	      actionInputs[i*nSplits+j]=aiSplit[j];
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
  torch::save(stateInputs.slice(0,nTr,-1,1),path+"stateInputsTest.pt");
  torch::save(normalizeActions(actionInputs).slice(0,nTr,-1,1),path+"actionInputsTest.pt");
  torch::save(rewardLabels.slice(0,nTr,-1,1),path+"rewardLabelsTest.pt");
  torch::save(stateLabels.slice(0,nTr,-1,1),path+"stateLabelsTest.pt");  
}

float ToolsSS::comparePosMSE(torch::Tensor initState, int nWaypoints, torch::Tensor actionSequence, torch::Tensor estimate)
{
  sw = SpaceWorld(tensorToVector(initState),nWaypoints);
  int T = actionSequence.size(0);
  torch::Tensor labels = torch::zeros({T,initState.size(0)});
  for (int t=0;t<T;t++)
    {
      sw.setTakenAction(tensorToVector(actionSequence[t]));
      sw.transition();
      labels[t] = torch::tensor(sw.getCurrentState().getStateVector());
    }
  return *moduloMSE(estimate.slice(-1,0,2,1),labels.slice(-1,0,2,1),false).data<float>();
}

void ToolsSS::generateSeed(int nTimesteps, int nRollouts, string filename)
{
  torch::Tensor actions = torch::zeros(0);
  //  actions = torch::cat({actions,torch::zeros({nTimesteps,nRollouts,4}).normal_(0,1)},2);
  actions = torch::cat({actions,torch::zeros({nTimesteps,nRollouts,4}).normal_(0,1000)},2); 
  for (unsigned int i=0;i<2;i++)
    {
      torch::Tensor center = torch::rand({nRollouts});
      torch::Tensor initCA = torch::zeros({nRollouts,nTimesteps,1});
      for (int k=0;k<nRollouts;k++)
	{
	  initCA[k] = torch::clamp(torch::zeros({nTimesteps,1}).normal_(*center[k].data<float>(),0.1),0,1);
	}      
      actions = torch::cat({actions,initCA.transpose(0,1)},2);
    }
  cout<<actions<<endl;
  torch::save(actions,filename);
}

void ToolsSS::transitionAccuracy(torch::Tensor testData, torch::Tensor labels, int nSplit, bool disp)
{
  int s = testData.size(1);
  int n = testData.size(0);
  pMSE+=moduloMSE(testData.slice(1,0,2,1),labels.slice(1,0,2,1),false)/nSplit;
  vMSE+=torch::mse_loss(testData.slice(1,2,4,1),labels.slice(1,2,4,1))/nSplit;  
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
  int m = testData.size(0);
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
