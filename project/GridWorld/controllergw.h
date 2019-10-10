#ifndef CONTROLLERGW_H
#define CONTROLLERGW_H
#include "controller.h"
#include "mapgw.h"

#define WIN_REWARD 1
#define LOSE_REWARD -1
#define EMPTY_SQUARE_REWARD 0

class ControllerGW: public Controller
{
public:
    ControllerGW();
    ControllerGW(string mapTag);
    ControllerGW(string mapTag, double agentXInit, double agentYInit);
    double transition();
    bool isTerminal(State s);
    void updateStateVector();
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();

    int getSize() const;

private:
    int size;
    bool randomStart;
    vector<vector<double>> obstacles;
    double initX;
    double initY;
    double agentX;
    double agentY;
    double previousAgentX;
    double previousAgentY;
    double goalX;
    double goalY;
};

#endif // CONTROLLERGW_H
