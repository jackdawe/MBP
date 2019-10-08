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
    vector<int> accessibleStates();

    double getAgentX() const;
    void setAgentX(double value);
    double getAgentY() const;
    void setAgentY(double value);
    double getGoalX() const;
    double getGoalY() const;
    int getSize() const;

private:
    int size;
    vector<vector<double>> obstacles;
    double agentX;
    double agentY;
    double goalX;
    double goalY;
};

#endif // STATEGW_H
