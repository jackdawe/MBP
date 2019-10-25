#ifndef GRIDWORLD_H
#define GRIDWORLD_H
#include "world.h"
#include "mapgw.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#define WIN_REWARD 1
#define LOSE_REWARD -1
#define EMPTY_SQUARE_REWARD 0

class GridWorld: public World
{
public:
    GridWorld();
    GridWorld(string filename, bool getImageMode = false);
    GridWorld(string filename, float agentXInit, float agentYInit, bool getImageMode = false);
    void init(string mapTag);
    float transition();
    bool isTerminal(State s);
    void generateVectorStates();
    cv::Mat toRGBMat(vector<float> stateVector);
    torch::Tensor toRGBTensor(vector<float> stateVector);
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();

private:
    bool imageMode;
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
