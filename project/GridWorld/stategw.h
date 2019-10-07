#ifndef STATEGW_H
#define STATEGW_H
#include "state.h"
#include "mapgw.h"

class StateGW: public State
{
public:
    StateGW();
    StateGW(MapGW *map, int agentX, int agentY);
    void transition(Action a);
    double reward(Action a);
    bool isTerminal();
    void generateStateVector();

    int getAgentX() const;
    void setAgentX(int value);
    int getAgentY() const;
    void setAgentY(int value);
    int getGoalX() const;
    int getGoalY() const;

private:
    vector<vector<int>> obstacles;
    int agentX;
    int agentY;
    int goalX;
    int goalY;
};

#endif // STATEGW_H
