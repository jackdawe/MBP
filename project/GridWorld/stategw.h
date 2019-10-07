#ifndef STATEGW_H
#define STATEGW_H
#include "state.h"
#include "mapgw.h"

#define WIN_REWARD 1
#define LOSE_REWARD -1
#define EMPTY_SQUARE_REWARD 0

class StateGW: public State
{
public:
    StateGW();
    StateGW(MapGW map);
    void transition(double a);
    double reward(double a);
    bool isTerminal();
    void generateStateVector();

    int getAgentX() const;
    void setAgentX(int value);
    int getAgentY() const;
    void setAgentY(int value);
    int getGoalX() const;
    int getGoalY() const;
    int getSize() const;

private:
    int size;
    vector<vector<int>> obstacles;
    int agentX;
    int agentY;
    int goalX;
    int goalY;
};

#endif // STATEGW_H
