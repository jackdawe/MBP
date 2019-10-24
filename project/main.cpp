#include <QApplication>
#include "GridWorld/gridworld.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"
int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);

    string mapTag = "2_8";
    GridWorld gw(mapTag);
    int size = gw.getSize();
    ConvNetGW net(size,16,32,size*size*2);
//    float gamma = stof(argv[1]);
//    float learningRate = stof(argv[2]);
//    float entropyMul = stof(argv[3]);
//    int batchSize = stoi(argv[4]);
//    int nEpisodes = stoi(argv[5]);
    float gamma = 0.95;
    float learningRate = 0.003;
    float entropyMul = 0.05;
    int batchSize = 10;
    int nEpisodes = 2000;
    ParametersA2C params(gamma,learningRate,entropyMul,batchSize,nEpisodes);
    ActorCritic<GridWorld,ConvNetGW> agent(gw,net,params,true);
    agent.train();
    agent.saveTrainingData();
//    a.exec();
}


