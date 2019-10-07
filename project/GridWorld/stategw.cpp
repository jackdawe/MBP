#include "stategw.h"

StateGW::StateGW()
{

}

StateGW::StateGW(MapGW map)
{
    default_random_engine generator(std::random_device{}());
    uniform_int_distribution<int> dist(1,map.getSize()-1);

    for (int i=0;i<map.getSize();i++)
    {
        obstacles.push_back(vector<int>(map.getSize(),0));
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

void StateGW::transition(Action a)
{

}

double StateGW::reward(Action a)
{

}

bool StateGW::isTerminal()
{

}

void StateGW::generateStateVector()
{

}

int StateGW::getAgentX() const
{
    return agentX;
}

void StateGW::setAgentX(int value)
{
    agentX = value;
}

int StateGW::getAgentY() const
{
    return agentY;
}

void StateGW::setAgentY(int value)
{
    agentY = value;
}

int StateGW::getGoalX() const
{
    return goalX;
}

int StateGW::getGoalY() const
{
    return goalY;
}

int StateGW::getSize() const
{
    return size;
}

