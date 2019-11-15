#ifndef GRIDWORLD_H
#define GRIDWORLD_H
#include "world.h"
#include "mapgw.h"
#include <opencv2/opencv.hpp>
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS

#define WIN_REWARD 1
#define LOSE_REWARD -1
#define EMPTY_SQUARE_REWARD -0.01;

class GridWorld: public World
{
public:
    GridWorld();
    GridWorld(string filename);
    GridWorld(string filename, float agentXInit, float agentYInit);
    GridWorld(string mapPoolPath, int mapPoolSize);
    void init();
    float transition();
    bool isTerminal(State s);
    void generateVectorStates();
    cv::Mat toRGBMat(vector<float> stateVector);
    torch::Tensor toRGBTensor(vector<float> stateVector);
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();
    void generateDataSet(int n);
    
    
private:
    MapGW map;
    bool randomStart;
    string mapTag;
    int mapPoolSize;
    vector<vector<float>> obstacles;
    float initX;
    float initY;
    float agentX;
    float agentY;
    float goalX;
    float goalY;
};

#endif // GRIDWORLD_H
