#include "gridworld.h"

GridWorld::GridWorld(): imageMode(imageMode)
{
}

GridWorld::GridWorld(string mapTag, bool imageMode): imageMode(imageMode)
{
    randomStart = true;
    init(mapTag);
    default_random_engine generator(std::random_device{}());
    uniform_int_distribution<int> dist(1,size-1);
    agentX = dist(generator), agentY = dist(generator);
    while ((agentX == goalX && agentY == goalY) || obstacles[agentX][agentY] == 1)
    {
        agentX = dist(generator), agentY = dist(generator);
    }
}

GridWorld::GridWorld(string mapTag, float agentXInit, float agentYInit, bool imageMode):
    imageMode(imageMode),initX(agentXInit), initY(agentYInit), agentX(agentXInit),agentY(agentYInit)
{
    randomStart=false;
    init(mapTag);
}

void GridWorld::init(string mapTag)
{
    MapGW map;
    map.load(mapTag);
    vector<DiscreteAction> dactions = {DiscreteAction(4)};
    actions = ActionSpace(dactions, vector<ContinuousAction>());
    rewardHistory.push_back(0);
    takenAction = vector<float>(1,0);
    size = map.getSize();
    path = "../GridWorld/Map"+mapTag+"/";
    for (int i=0;i<size;i++)
    {
        obstacles.push_back(vector<float>(map.getSize(),0));
    }

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
            }
        }
    }
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
    rewardHistory.back()+= r;//*(actionSequence.size());
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
    currentState.add(agentX),currentState.add(agentY),
            currentState.add(goalX), currentState.add(goalY);
    for (int i=0;i<size;i++)
    {
        for (int j=0;j<size;j++)
        {
            currentState.add(obstacles[i][j]);
        }
    }
    previousState = State(currentState);
    stateSequence.push_back(currentState.getStateVector());
}

cv::Mat GridWorld::toRGBMat(State s)
{
    vector<float> stateVector = s.getStateVector();
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

torch::Tensor GridWorld::toRGBTensor(State s)
{
    cv::Mat m = toRGBMat(s);
    m.convertTo(m,CV_32FC3,1.0f/255.0f);
    torch::Tensor tensorS = torch::from_blob(m.data,{1,m.rows,m.cols,3});
    return tensorS.permute({0,3,1,2});
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
    if (randomStart)
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
    currentState.update(0,agentX), currentState.update(1,agentY);
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

int GridWorld::getSize() const
{
    return size;
}
