#include <QApplication>
//#include "actionspace.h"
#include "GridWorld/controllergw.h"
//#include "agenttrainer.h"
//#include "Agents/qlearning.h"
//#include "Starship/episodeplayerss.h"
#include "Agents/A2C/actorcritic.h"

int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);

    string mapTag = "2_8";
    ControllerGW c(mapTag);
    ParametersA2C params(0.95,0.003,0.01,5,10000,{64,128,64});
    ActorCritic<ControllerGW,ModelA2CGW> agent(c,params);
    agent.train();
    ControllerGW c2(mapTag,5,2);
    c2.generateStates();
    cout<< agent.getModel().actorOutput(torch::tensor(c2.getCurrentState().getStateVector()).reshape({1,68})) <<endl;
//    ModelA2CGW model(4,64,128,64);
//    float a = 2.2;
//    torch::Tensor x = torch::tensor({a,a,a,a}).view({-1,1});
//    x = model.actorOutput(x);
//    cout<<x<<endl;
//    return a.exec();

}
