#ifndef GRIDWORLD_H
#define GRIDWORLD_H
#include "world.h"
#include "mapgw.h"
#include <opencv2/opencv.hpp>

#define WIN_REWARD 1
#define LOSE_REWARD -1
#define EMPTY_SQUARE_REWARD 0

class GridWorld: public World
{
public:
    GridWorld();
    GridWorld(string mapTag, bool getImageMode = false);
    GridWorld(string mapTag, float agentXInit, float agentYInit, bool getImageMode = false);
    void init(string mapTag);
    float transition();
    bool isTerminal(State s);
    void generateVectorStates();
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();

    int getSize() const;

private:
    bool imageMode;
    int size;
    bool randomStart;
    vector<vector<float>> obstacles;
    float initX;
    float initY;
    float agentX;
    float agentY;
    float goalX;
    float goalY;
};

#endif // GRIDWORLD_H
