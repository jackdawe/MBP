#ifndef CONTROLLERGW_H
#define CONTROLLERGW_H
#include "controller.h"
#include "mapgw.h"
#include <opencv2/opencv.hpp>

#define WIN_REWARD 1
#define LOSE_REWARD -1
#define EMPTY_SQUARE_REWARD 0

class ControllerGW: public Controller
{
public:
    ControllerGW();
    ControllerGW(string mapTag, bool getImageMode = false);
    ControllerGW(string mapTag, float agentXInit, float agentYInit, bool getImageMode = false);
    void init(string mapTag);
    float transition();
    bool isTerminal(State s);
    void generateVectorStates();
    void generateImageStates();
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
    float goalX;
    float goalY;
};

#endif // CONTROLLERGW_H
