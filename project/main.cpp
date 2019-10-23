#include <QApplication>
//#include "actionspace.h"
#include "GridWorld/controllergw.h"
//#include "agenttrainer.h"
//#include "Agents/qlearning.h"
//#include "Starship/episodeplayerss.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    string mapTag = "Test";
    ControllerGW c(mapTag);
    ParametersA2C params(0.95,0.003,0.01,5,10000,{64,128,64});
    ActorCritic<ControllerGW,ModelA2CGW> agent(c,params);
    agent.train();

//    agent.playOne();
//    c.loadEpisode("A2C_MLP_65");
//    EpisodePlayerGW ep(mapTag,c.getStateSequence());
//    ep.playEpisode();
//    a.exec();
//    ControllerGW c2(mapTag,6,1);
//    c2.generateStates();
//    cout<< agent.getModel().actorOutput(torch::tensor(c2.getCurrentState().getStateVector()).reshape({1,68})) <<endl;

//    QApplication a(argc, argv);
//    string mapTag = "Test";
//    ControllerGW c(mapTag);
//    ParametersA2C params(0.95,0.003,0.0001,10,5000,{64,128,64});
//    ActorCritic<ControllerGW,ModelA2CGW> agent(c,params);
//    agent.train();
//    vector<vector<float>> r;
//    for (int i=0;i<c.getSize();i++)
//    {
//        vector<float> rl;
//        for (int j=0;j<c.getSize();j++)
//        {
//            c = ControllerGW(mapTag,i,j);
//            c.generateStates();
//            agent.setController(c);
//            rl.push_back(*agent.getModel().criticOutput(torch::tensor(c.getCurrentState().getStateVector()).reshape({1,68})).data<float>());
//        }
//        r.push_back(rl);
//    }
//    EpisodePlayerGW ep(mapTag);
//    ep.showScores(r);
//    a.exec();
}


