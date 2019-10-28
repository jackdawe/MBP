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
    int b;
    //LOADING MAP AND TRAINING AGENT

    string mapTag = "../GridWorld/MapPools/8x8/Easy/Train/map1";
    GridWorld gw(mapTag,true);
    int size = gw.getSize();
    ConvNetGW net(size,16,32,size*size*2);

//    float gamma = 0.90;
//    float learningRate = 0.003;
//    float beta = 0.01;
//    float zeta = 0.5;
//    int batchSize = 100;
//    int nEpisodes = 20000;
    ParametersA2C params(gamma,learningRate,beta,zeta,batchSize,nEpisodes);
    ActorCritic<GridWorld,ConvNetGW> agent(gw,net,params,true);
    agent.train();
    agent.saveTrainingData();

    //SHOWING THE POLICY

    vector<vector<string>> texts;
    for (int i=0;i<size;i++)
    {
        vector<string> textsL(size,"");
        for (int j=0;j<size;j++)
        {
            GridWorld gw2(mapTag,i,j);
            gw2.generateVectorStates();
            torch::Tensor output = agent.getModel().actorOutput(gw2.toRGBTensor(gw2.getCurrentState().getStateVector()));
            float max = 0;
            int dir;
            string sdir;
            for (int k=0;k<4;k++)
            {
                if (*output[0][k].data<float>()>max)
                {
                    max = *output[0][k].data<float>();
                    dir = k;
                }
            }
            switch (dir)
            {
            case 0:
                sdir = "UP";
                break;
            case 1:
                sdir = "RIGHT";
                break;
            case 2:
                sdir = "DOWN";
                break;
            case 3:
                sdir = "LEFT";
                break;
            }
            textsL[j] = sdir;
        }
        texts.push_back(textsL);
    }
    EpisodePlayerGW ep(mapTag);
    ep.displayOnGrid(texts);

    a.exec();
}

//    int nMaps = 10;
//    default_random_engine g(random_device{}());
//    uniform_int_distribution<int> dist(1,3);
//    for (int i=0;i<nMaps;i++)
//    {
//        string filename = "../GridWorld/MapPools/8x8/Easy/Train/map"+to_string(i);
//        MapGW map(8);
//        map.generate(dist(g));
//        map.save(filename);
//    }


//vector<torch::Tensor> batches;
//torch::Tensor batch;
//default_random_engine gen(random_device{}());
//uniform_int_distribution<int> dist(1,7);
//vector<torch::Tensor> lBatches;
//torch::Tensor lBatch;
//torch::optim::Adam optimizer(net.parameters(),0.003);
//for (int i=0;i<101;i++)
//{
//    batch = torch::zeros({50,3,8,8});
//    lBatch = torch::zeros({50,4});
//    for (int j=0;j<50;j++)
//    {
//        int x = dist(gen);
//        int y = dist(gen);
//        if (x == 1)
//        {
//            lBatch[j][3] = 1;
//        }
//        else
//        {
//            lBatch[j][0] = 1;
//        }
//        GridWorld gw2(mapTag,x,y);
//        gw2.generateVectorStates();
//        batch[j] = gw2.toRGBTensor(gw2.getCurrentState().getStateVector()).reshape({3,8,8});
//    }
//    batches.push_back(batch);
//    lBatches.push_back(lBatch);
//}

//for (int i=0;i<100;i++)
//{
//    torch::Tensor output = net.actorOutput(batches[i]);
//    torch::Tensor loss = torch::binary_cross_entropy(output,lBatches[i]);
//    optimizer.zero_grad();
//    loss.backward();
//    optimizer.step();
//}

//torch::Tensor output = net.actorOutput(batches[100]);
//cout<<lBatches[100]<<endl;
//cout<<output<<endl;
