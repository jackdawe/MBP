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

    string mapTag = "2_8";
    ControllerGW c(mapTag);
    ParametersA2C params(0.95,0.003,0.005,10,5000,{64,128,64});
    ActorCritic<ControllerGW,ModelA2CGW> agent(c,params);
    agent.train();


//    agent.playOne();
//    ControllerGW c(mapTag);
//    c.loadEpisode("A2C_MLP_2883");
//    EpisodePlayerGW ep(mapTag,c.getStateSequence());
//    ep.playEpisode();
//    a.exec();
    ControllerGW c2(mapTag,6,1);
    c2.generateStates();
    cout<< agent.getModel().actorOutput(torch::tensor(c2.getCurrentState().getStateVector()).reshape({1,68})) <<endl;
}
