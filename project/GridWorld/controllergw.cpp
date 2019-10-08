#include "controllergw.h"

ControllerGW::ControllerGW()
{

}

ControllerGW::ControllerGW(MapGW map)
{
    generateStateVector();
    vector<DiscreteAction> dactions = {DiscreteAction(4)};
    actions = ActionSpace(dactions, vector<ContinuousAction>());
    default_random_engine generator(std::random_device{}());
    uniform_int_distribution<int> dist(1,map.getSize()-1);
    size = map.getSize();
    for (int i=0;i<size;i++)
    {
        obstacles.push_back(vector<double>(map.getSize(),0));
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
    agentX = dist(generator), agentY = dist(generator);
    while ((agentX == goalX && agentY == goalY) || obstacles[agentX][agentY] == 1)
    {
        agentX = dist(generator), agentY = dist(generator);
    }
}

double ControllerGW::transition(double a)
{
    if (!isTerminal())
    {
        switch ((int)a)
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
    }
    if (obstacles[agentX][agentY] == 1)
    {
        return LOSE_REWARD;
    }
    if (agentX == goalX && agentY == goalY)
    {
        return WIN_REWARD;
    }
    return EMPTY_SQUARE_REWARD;
}

bool ControllerGW::isTerminal()
{
    return obstacles[agentX][agentY] == 1 || (agentX == goalX && agentY == goalY);
}

void ControllerGW::generateStateVector()
{
    currentState.add(&agentX),currentState.add(&agentY),
            currentState.add(&goalX), currentState.add(&goalY);
    for (unsigned int i=0;i<size;i++)
    {
        for (unsigned int j=0;j<size;j++)
        {
            currentState.add(&obstacles[i][j]);
        }
    }
}

int ControllerGW::stateId(State s)
{
    double ax = *s.getStateVector()[0];
    double ay = *s.getStateVector()[1];
    return ax*size+ay;
}

vector<int> ControllerGW::accessibleStates(State s)
{
    double ax = *s.getStateVector()[0];
    double ay = *s.getStateVector()[1];
    vector<int> accessibleStates = {(ax-1)*size+ay,ax*size+ay+1,(ax+1)*size+ay,ax*size+ay-1};
    return accessibleStates;
}

int ControllerGW::spaceStateSize()
{
    return size*size;
}

double ControllerGW::getAgentX() const
{
    return agentX;
}

void ControllerGW::setAgentX(double value)
{
    agentX = value;
}

double ControllerGW::getAgentY() const
{
    return agentY;
}

void ControllerGW::setAgentY(double value)
{
    agentY = value;
}

double ControllerGW::getGoalX() const
{
    return goalX;
}

double ControllerGW::getGoalY() const
{
    return goalY;
}

int ControllerGW::getSize() const
{
    return size;
}
