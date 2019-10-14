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
    ControllerGW(string mapTag, float agentXInit, float agentYInit);
    float transition();
    bool isTerminal(State s);
    void generateStates();
    void updateStates();
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();

    int getSize() const;

private:
    int size;
    bool randomStart;
    vector<vector<float>> obstacles;
    float initX;
    float initY;
    float agentX;
    float agentY;
    float previousAgentX;
    float previousAgentY;
    float goalX;
    float goalY;
};

#endif // CONTROLLERGW_H
