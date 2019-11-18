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
  return rgbState.reshape({1,3,size,size});
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
  if (mapPoolSize != -1)
    {
      default_random_engine generator(std::random_device{}());
      uniform_int_distribution<int> dist(0,mapPoolSize-1);
      int mapId = dist(generator); 
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
  //Initialising the tensors that will contain the dataset
  
  torch::Tensor stateInputs = torch::zeros(0);
  torch::Tensor actionInputs = torch::zeros(0);
  torch::Tensor stateLabels = torch::zeros(0);
  torch::Tensor rewardLabels = torch::zeros(0);
  
  //Making the agent wander randomly for n episodes 
  
  for (int i=0;i<n;i++)
    {
      //Displaying a progression bar in the terminal
      
      if (n > 100 && i%(5*n/100) == 0)
	{
	  cout << "Your agent is crashing into walls for science... " + to_string(i/(n/100)) + "%" << endl;
	}

      if (i==4.*n/5)
	{
	  cout<< "Your training set is complete and contains " + to_string(stateInputs.size(0)) +" elements! Now generating test set..."<<endl; 
	  torch::save(stateInputs,path+"../stateInputsTrain.pt");
	  torch::save(actionInputs,path+"../actionInputsTrain.pt");
	  torch::save(rewardLabels,path+"../rewardLabelsTrain.pt");
	  torch::save(stateLabels,path+"../stateLabelsTrain.pt");
	  stateInputs = torch::zeros(0);
	  actionInputs = torch::zeros(0);
	  stateLabels = torch::zeros(0);
	  rewardLabels = torch::zeros(0);
	}
      
      while(!isTerminal(currentState))
	{
	  torch::Tensor s = toRGBTensor(currentState.getStateVector());
	  stateInputs = torch::cat({stateInputs, s}); 
	  torch::Tensor action = torch::zeros({1});
	  action[0] = randomAction()[0]; 
	  actionInputs = torch::cat({actionInputs,action});
	  torch::Tensor reward = torch::zeros({1});
	  reward[0] = transition();
	  rewardLabels = torch::cat({rewardLabels,reward});
	  s = toRGBTensor(currentState.getStateVector());
	  stateLabels = torch::cat({stateLabels,s});
	}
      reset();
    }
  
  //Saving the model
  
  cout<< "Your test set is complete and contains " + to_string(stateInputs.size(0)) +" elements!"<<endl; 
  
  torch::save(stateInputs,"../stateInputsTest.pt");
  torch::save(actionInputs,"../actionInputsTest.pt");
  torch::save(rewardLabels,"../rewardLabelsTest.pt");
  torch::save(stateLabels,"../stateLabelsTest.pt");
  
}

torch::Tensor GridWorld::predictionToRGBState(torch::Tensor testData, torch::Tensor labels)
{
  int n = testData.size(2);
  int m = testData.size(0);
  testData = testData.to(torch::Device(torch::kCPU));
  vector<vector<int>> scores;
  vector<vector<int>> truth;
  for (int i=0;i<3;i++)
    {
      scores.push_back(vector<int>(4,0));
      truth.push_back(vector<int>(2,0));
    }
  for (int s=0;s<m;s++)
    {
      for (int i=0;i<n;i++)
	{
	  for (int j=0;j<n;j++)
	    {
	      for (int c=0;c<3;c++)
		{
		  float pixelt = *testData[s][c][i][j].data<float>();
		  float pixell = *labels[s][c][i][j].data<float>();
		  if (pixelt>0.5 && pixell<1.2)
		    {
		      testData[s][c][i][j] = 1;
		      pixelt=1;
		    }
		  else
		    {
		      testData[s][c][i][j] = 0;
		      pixelt=0;
		    }
		  if (pixelt == 1)
		    {		      
		      if (pixell == 1)
			{
			  truth[c][1]++;
			  scores[c][0]++;
			}
		      else
			{
			  truth[c][0] ++;
			  scores[c][1]++;
			}
		    }
		  else
		    {		      
		      if(pixell == 1)
			{
			  truth[c][1]++;
			  scores[c][2]++;
			}
		      else
			{
			  truth[c][0]++;
			  scores[c][3]++;
			}		      
		    }
		}
	    }
	}
    }
  cout<<"Performance on the test set:"<<endl;
  vector<string> names = {"Channel 1 - Agent","Channel 2 - Goal", "Channel 3 - Obstacles"};
  vector<string> names2 = {"True Positives","False Positives","False Negatives","True Negatives"};
  
  for (int i=0;i<3;i++)
    {
      cout<<names[i]+":"<<endl;
      for (int j=0;j<4;j++)
	{
	  int idx = 0;
	  if (j<2)
	    {
	      idx = 1;
	    }
	  cout<<names2[j] + ": " + to_string(100.*scores[i][j]/(m*n*n)) + "% " + to_string(scores[i][j])+"/"+to_string(truth[i][idx]) << endl;
	}
    }
  return testData;
}
