#include "toolsgw.h"
DEFINE_double(wp,0.1,"Percentage of forced win scenarios during the dataset generation"); 

ToolsGW::ToolsGW(){}

ToolsGW::ToolsGW(GridWorld gw): gw(gw){}

void ToolsGW::generateDataSet(string path, int nmaps, int n, float winProp)
{
  gw = GridWorld(path+"train/",nmaps);
  gw.generateVectorStates();
  
  //Initialising the tensors that will contain the training set

  int size = gw.getSize();
  torch::Tensor stateInputs = torch::zeros({4*n/5,3,size,size});
  torch::Tensor actionInputs = torch::zeros({4*n/5});
  torch::Tensor stateLabels = torch::zeros({4*n/5,size,size});
  torch::Tensor rewardLabels = torch::zeros({4*n/5});
  
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
	  stateInputs = torch::zeros({n/5,3,size,size});
	  actionInputs = torch::zeros({n/5});
	  stateLabels = torch::zeros({n/5,size,size});
	  rewardLabels = torch::zeros({n/5});
	}

      //Building the dataset tensors
      
      torch::Tensor s = gw.toRGBTensor(gw.getCurrentState().getStateVector());
      stateInputs[j] = s;
      if (i>winProp*4*n/5 && i<n-winProp*n/5)
	{
	  gw.setTakenAction(gw.randomAction()); //Not changing the action when in winning scenarios generation
	}
      actionInputs[j] = gw.getTakenAction()[0];
      float r = gw.transition();
      float bug = EMPTY_SQUARE_REWARD; //Code does not compile if I use EMPTY_SQUARE_REWARD in switch
      if(r == LOSE_REWARD)
	{
	  r = 0;
	}
      else if (r == bug)
	{
	  r = 1;
	}
      else if (r == WIN_REWARD)
	{
	  r = 2;
	}
      rewardLabels[j] = r;
      s = gw.toRGBTensor(gw.getCurrentState().getStateVector())[0];
      stateLabels[j] = s;

      //Resettiing the gridworld at the end of an episode
      
      if (gw.isTerminal(gw.getCurrentState()))
	{
	  gw.reset();
	  
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
      j++;
    }
  

  //Saving the model
  
  cout<< "Test set generation is complete!"<<endl; 
  
  torch::save(stateInputs,path+"stateInputsTest.pt");
  torch::save(actionInputs,path+"actionInputsTest.pt");
  torch::save(rewardLabels,path+"rewardLabelsTest.pt");
  torch::save(stateLabels,path+"stateLabelsTest.pt");
}

void ToolsGW::transitionAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  cout<<"ee"<<endl;
  
  int n = testData.size(2);
  int m = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));
  vector<int> scores(2,0);
  vector<int> truth(2,0);
  cout<<"hello"<<endl;
  for (int s=0;s<m;s++)
    {      
      for (int i=0;i<n;i++)
	{
	  for (int j=0;j<n;j++)
	    {
	      float pixelt = *testData[s][i][j].data<float>();
	      float pixell = *labels[s][i][j].data<float>();
		if (pixelt>0.9 && pixell<1.1)
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
  int m = testData.size(0);
  vector<int> rCounts(3,0);
  vector<int> scores(3,0);
  testData = torch::argmax(torch::exp(testData),1);
  testData = testData.to(torch::Device(torch::kCPU));
  labels = labels.to(torch::kInt32);
  for (int s=0;s<m;s++)
    {
      int rl = *labels[s].data<int>();
      rCounts[rl]++;
      if (*testData[s].data<long>() == rl)
	{
	  scores[rl]++;
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
