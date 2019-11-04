#include <QApplication>
#include "GridWorld/gridworld.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"

int main(int argc, char *argv[])
{
    float gamma = stof(argv[1]);
    float learningRate = stof(argv[2]);
    float beta = stof(argv[3]);
    float zeta = stof(argv[4]);
    int batchSize = stoi(argv[5]);
    int nEpisodes = stoi(argv[6]);
    QApplication a(argc, argv);
    //LOADING MAP AND TRAINING AGENT
//    string mapTag = "../GridWorld/Map1_8/map";
//    GridWorld gw(mapTag,true);
//    int size = gw.getSize();
//    ConvNetGW net(size,16,16,size*size*2);

//    float gamma = 0.99;
//    float learningRate = 0.003;
//    float beta = 0.001;
//    float zeta = 0.5;
//    int batchSize = 100;
//    int nEpisodes = 20000;
//    ParametersA2C params(gamma,learningRate,beta,zeta,batchSize,nEpisodes);
//    ActorCritic<GridWorld,ConvNetGW> agent(gw,net,params,true);
//    agent.train();
//    agent.saveTrainingData();

    MapGW map(8);
    map.generateMapPool(10,"../GridWorld/MapPools/8x8/Intermediate/Training/",100);


    //SHOWING THE POLICY

//    vector<vector<string>> texts;
//    vector<vector<string>> texts2;
//    for (int i=0;i<size;i++)
//    {
//        vector<string> textsL(size,"");
//        vector<string> textsL2(size,"");
//        for (int j=0;j<size;j++)
//        {
//            GridWorld gw2(mapTag,i,j);
//            gw2.generateVectorStates();
//            torch::Tensor output = agent.getModel().actorOutput(gw2.toRGBTensor(gw2.getCurrentState().getStateVector()));
//            torch::Tensor output2 = agent.getModel().criticOutput(gw2.toRGBTensor(gw2.getCurrentState().getStateVector()));
//            float max = 0;
//            int dir;
//            string sdir;
//            for (int k=0;k<4;k++)
//            {
//                if (*output[0][k].data<float>()>max)
//                {
//                    max = *output[0][k].data<float>();
//                    dir = k;
//                }
//            }
//            switch (dir)
//            {
//            case 0:
//                sdir = "UP";
//                break;
//            case 1:
//                sdir = "RIGHT";
//                break;
//            case 2:
//                sdir = "DOWN";
//                break;
//            case 3:
//                sdir = "LEFT";
//                break;
//            }
//            textsL[j] = sdir;

//            string val = to_string(*output2.data<float>());
//            string val2;
//            val2+=val[0],val2+=val[1],val2+=val[2],val2+=val[3],val2+=val[4];
//            textsL2[j] = val2;
//        }
//        texts.push_back(textsL);
//        texts2.push_back(textsL2);
//    }
//    EpisodePlayerGW ep(mapTag);
//    ep.displayOnGrid(texts);

    a.exec();
}
