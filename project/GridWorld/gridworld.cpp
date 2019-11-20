#include "gridworld.h"

GridWorld::GridWorld(){}

GridWorld::GridWorld(string filename):
  mapTag(filename), randomStart(true), mapPoolSize(-1)
{
  init();
}

GridWorld::GridWorld(string filename, float agentXInit, float agentYInit):
  mapTag(filename), randomStart(false),initX(agentXInit), initY(agentYInit), agentX(agentXInit),agentY(agentYInit),mapPoolSize(-1)
{
  init();
}

GridWorld::GridWorld(string mapPoolPath, int mapPoolSize):
  mapTag(mapPoolPath), randomStart(true), mapPoolSize(mapPoolSize)
{
  init();
}

void GridWorld::init()
{
  vector<DiscreteAction> dactions = {DiscreteAction(4)};
  actions = ActionSpace(dactions, vector<ContinuousAction>());
  takenAction = vector<float>(1,0);
  if (mapPoolSize == -1)
    {
      map.load(mapTag);
    }
  else
    {
      map.load(mapTag+"map0");
    }
  size = map.getSize();
  for (int i=0;i<4+size*size;i++)
    {
      currentState.add(0);
    }
  for (int i=0;i<size;i++)
    {
      obstacles.push_back(vector<float>(map.getSize(),0));
    }
  reset();
}

float GridWorld::transition()
{
    int a = (int)takenAction[0];
    float r = 0;
    previousState.update(0,agentX), previousState.update(1,agentY);
    if (!isTerminal(currentState))
    {
        switch (a)
        {
        case 0:
            agentX--;
            break;
        case 1:
            agentY++;
            break;
        case 2:
            agentX++;
            break;
        case 3:
            agentY--;
            break;
        }
        currentState.update(0,agentX), currentState.update(1,agentY);
        actionSequence.push_back({a});
        stateSequence.push_back(currentState.getStateVector());
    }   
    if (obstacles[agentX][agentY] == 1)
    {
        r = LOSE_REWARD;
    }
    else if (agentX == goalX && agentY == goalY)
    {
        r = WIN_REWARD;
    }
    else
    {
        r = EMPTY_SQUARE_REWARD;
    }
    rewardHistory.back()+= r;
    return r;
}

bool GridWorld::isTerminal(State s)
{
    float ax = s.getStateVector()[0];
    float ay = s.getStateVector()[1];
    return obstacles[ax][ay] == 1 || (ax == goalX && ay == goalY);
}

void GridWorld::generateVectorStates()
{
    currentState.update(0,agentX),currentState.update(1,agentY),
            currentState.update(2,goalX), currentState.update(3,goalY);
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            currentState.update(4+i*size+j,obstacles[i][j]);
        }
    }
    previousState = State(currentState);
    stateSequence.push_back(currentState.getStateVector());
}

cv::Mat GridWorld::toRGBMat(vector<float> stateVector)
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
}

torch::Tensor GridWorld::toRGBTensor(vector<float> stateVector)
{
  torch::Tensor rgbState = torch::zeros({3,size,size});
  for (int i=0;i<size;i++)
    {
      for (int j=0;j<size;j++)
	{
	  rgbState[2][i][j] = stateVector[i*size+j+4];
	}
    }
  rgbState[0][stateVector[0]][stateVector[1]] = 1;
  rgbState[1][stateVector[2]][stateVector[3]] = 1;
  return rgbState;
}

int GridWorld::stateId(State s)
{
    float ax = s.getStateVector()[0];
    float ay = s.getStateVector()[1];
    return ax*size+ay;
}

void GridWorld::reset()
{  
  rewardHistory.push_back(0);
  default_random_engine generator(std::random_device{}());
  uniform_int_distribution<int> dist(0,mapPoolSize-1);
  int mapId; 
  if (mapPoolSize != -1)
    {
      default_random_engine generator(std::random_device{}());
      uniform_int_distribution<int> dist(0,mapPoolSize-1);
      mapId = dist(generator);
      map.load(mapTag+"map"+to_string(mapId));
      size = map.getSize();
    }
  if(obstacles[0][0]!=1 || mapPoolSize != -1)
    {
      for (int i=0;i<size;i++)
	{
	  for (int j=0;j<size;j++)
	    {
	      switch(map.getMap()[i][j])
		{
		case 1:
		  obstacles[i][j]=1;
		  break;
		case 2:
		  goalX=i, goalY=j;
		  break;
		default:
		  obstacles[i][j]=0;
		}
	    }
	}
    }
  if(randomStart)
    {
      default_random_engine generator(std::random_device{}());
      uniform_int_distribution<int> dist(1,size-1);
      agentX = dist(generator), agentY = dist(generator);
      while ((agentX == goalX && agentY == goalY) || obstacles[agentX][agentY] == 1)
	{
	  agentX = dist(generator), agentY = dist(generator);
	}
    }
  else
    {
      agentX = initX; agentY = initY;
    }
  generateVectorStates();
  actionSequence = vector<vector<float>>();
  stateSequence = {currentState.getStateVector()};
}

vector<int> GridWorld::accessibleStates(State s)
{
  int ax = s.getStateVector()[0];
  int ay = s.getStateVector()[1];
  vector<int> accessibleStates = {(ax-1)*size+ay,ax*size+ay+1,(ax+1)*size+ay,ax*size+ay-1};
  return accessibleStates;
}

int GridWorld::spaceStateSize()
{
  return size*size;
}

void GridWorld::generateDataSet(int n)
{
  //Initialising the tensors that will contain the training set
  
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

      if (i==4*n/5)
	{
	  j = 0;
	  cout<< "Training set generation is complete! Now generating test set..."<<endl; 
	  torch::save(stateInputs,path+"../stateInputsTrain.pt");
	  torch::save(actionInputs,path+"../actionInputsTrain.pt");
	  torch::save(rewardLabels,path+"../rewardLabelsTrain.pt");
	  torch::save(stateLabels,path+"../stateLabelsTrain.pt");
	  stateInputs = torch::zeros({n/5,3,size,size});
	  actionInputs = torch::zeros({n/5});
	  stateLabels = torch::zeros({n/5,size,size});
	  rewardLabels = torch::zeros({n/5});
	}
      torch::Tensor s = toRGBTensor(currentState.getStateVector());
      stateInputs[j] = s;      
      takenAction = randomAction();
      actionInputs[j] = takenAction[0];
      float r = transition();
      float bug = EMPTY_SQUARE_REWARD;
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
      s = toRGBTensor(currentState.getStateVector())[0];
      stateLabels[j] = s;
      if (isTerminal(currentState))
	{
	  //Adding win situations to the dataset as they occur rarely
	  reset();
	  if (i<8*n/25 || i>23*n/25)
	    {
	      vector<bool> available = {obstacles[goalX-1][goalY]==0 ,obstacles[goalX][goalY+1]==0,obstacles[goalX+1][goalY]==0,obstacles[goalX][goalY-1]==0};	      
	      
	      if (available == vector<bool>({false,false,false,false}))
		{
		  cout<<"This map has a trapped goal"<<endl;
		  reset();
		}
	      else
		{
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
		      agentX = goalX-1; agentY = goalY;
		      break;
		    case 1:
		      agentX = goalX; agentY = goalY+1;
		      break;
		    case 2:
		      agentX = goalX+1; agentY = goalY;
		      break;
		    case 3:
		      agentX = goalX; agentY = goalY-1;
		      break;
		    }
		  generateVectorStates();
		}
	    }
	}
      j++;
    }
  
  //Saving the model
  
  cout<< "Test set generation is complete!"<<endl; 
  
  torch::save(stateInputs,"../stateInputsTest.pt");
  torch::save(actionInputs,"../actionInputsTest.pt");
  torch::save(rewardLabels,"../rewardLabelsTest.pt");
  torch::save(stateLabels,"../stateLabelsTest.pt");
}

void GridWorld::transitionAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  int n = testData.size(2);
  int m = testData.size(0);
  testData = torch::chunk(testData,3,1)[0].reshape({m,n,n});
  testData = testData.to(torch::Device(torch::kCPU));
  vector<int> scores(4,0);
  vector<int> truth(4,0);
  
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
		if (pixelt == 1)
		  {		      
		    if (pixell == 1)
		      {
			truth[1]++;
			scores[0]++;
		      }
		    else
		      {
			truth[0]++;
			scores[1]++;
		      }
		  }
		else
		  {		      
		    if(pixell == 1)
		      {
			truth[1]++;
			scores[2]++;
		      }
		    else
		      {
			  truth[0]++;
			  scores[3]++;
			}		      
		  }
	    }
	}
    }
  cout<<"Performance of the transition function on the test set:"<<endl;
  vector<string> names = {"True Positives","False Positives","False Negatives","True Negatives"};
  for (int j=0;j<4;j++)
    {
      int idx = 0;
      if (j<2)
	{
	  idx = 1;
	}
      cout<<names[j] + ": " + to_string(100.*scores[j]/(m*n*n)) + "% " + to_string(scores[j])+"/"+to_string(truth[idx]) << endl;
    }
}


void GridWorld::rewardAccuracy(torch::Tensor testData, torch::Tensor labels)
{
  int m = testData.size(0);
  vector<int> rCounts(3,0);
  vector<int> scores(3,0);
  testData = torch::round(torch::exp(testData));
  testData = testData.to(torch::Device(torch::kCPU));
  labels = labels.to(torch::kInt32);
  for (int s=0;s<m;s++)
    {
      int rl = *labels[s].data<int>();
      rCounts[rl]++;
      if (*testData[s][rl].data<float>() == 1)
	{
	  scores[rl]++;
	}
    }
  vector<string> text = {"LOSE REWARD","EMPTY SQUARE REWARD","WIN REWARD"};
  cout<<"Performances of the reward function on the test maps: " << endl;
  for (int i=0;i<3;i++)
    {
      cout<<text[i]+": "+ to_string(scores[i]) + "/" + to_string(rCounts[i])<<endl;
    }
}
