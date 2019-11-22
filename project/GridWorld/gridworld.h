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
    GridWorld(string pathToMap);
    GridWorld(string pathToMap, float agentXInit, float agentYInit);
    GridWorld(string pathToDir, int mapPoolSize);
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
    void rewardAccuracy(torch::Tensor testData, torch::Tensor labels);

    vector<vector<float>> getObstacles();
    int getSize();
    float getAgentX();
    float getAgentY();
    float getGoalX();
    float getGoalY();
    void setAgentX(float x);
    void setAgentY(float y);
    void setGoalX(float x);
    void setGoalY(float y);
    
private:
    int size; //Map Size
    MapGW map;
    bool randomStart;
    string mapPoolPath;
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
