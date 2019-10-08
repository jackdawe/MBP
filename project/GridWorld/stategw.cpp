#include "stategw.h"

StateGW::StateGW()
{
    agentY = 1;
    generateStateVector();
    agentX = 1;
}

StateGW::StateGW(MapGW map)
{
    generateStateVector();
    default_random_engine generator(std::random_device{}());
    uniform_int_distribution<int> dist(1,map.getSize()-1);

    for (int i=0;i<map.getSize();i++)
    {
        obstacles.push_back(vector<double>(map.getSize(),0));
    }

    for (int i=0;i<map.getSize();i++)
    {
        for (int j=0;j<map.getSize();j++)
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

void StateGW::transition(double a)
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
}

double StateGW::reward(double a)
{
    StateGW s(*this);
    s.transition(a);
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

bool StateGW::isTerminal()
{
    return obstacles[agentX][agentY] == 1 || (agentX == goalX && agentY == goalY);
}

void StateGW::generateStateVector()
{
    stateVector.push_back(&agentX), stateVector.push_back(&agentY);
    stateVector.push_back(&goalX), stateVector.push_back(&goalY);
    for (unsigned int i=0;i<obstacles.size();i++)
    {
        for (unsigned int j=0;j<obstacles.size();j++)
        {
            stateVector.push_back(&obstacles[i][j]);
        }
    }
}

vector<int> StateGW::accessibleStates()
{
    unsigned int size = obstacles.size();
    vector<int> accessibleStates = {(agentX-1)*size+agentY,agentX*size+agentY+1,(agentX+1)*size+agentY,agentX*size+agentY-1};
    return accessibleStates;
}

double StateGW::getAgentX() const
{
    return agentX;
}

void StateGW::setAgentX(double value)
{
    agentX = value;
}

double StateGW::getAgentY() const
{
    return agentY;
}

void StateGW::setAgentY(double value)
{
    agentY = value;
}

double StateGW::getGoalX() const
{
    return goalX;
}

double StateGW::getGoalY() const
{
    return goalY;
}

int StateGW::getSize() const
{
    return size;
}

