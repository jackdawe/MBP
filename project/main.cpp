#include <QApplication>
//#include "actionspace.h"
#include "GridWorld/controllergw.h"
//#include "agenttrainer.h"
//#include "Agents/qlearning.h"
//#include "Starship/episodeplayerss.h"
#include "Agents/actorcritic.h"

int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);

    string mapTag = "2_8";
    ControllerGW c(mapTag);
    ModelA2CGW model(c.getSize()*c.getSize()+4,64,128,64);
    ActorCritic<ControllerGW,ModelA2CGW> agent(c,model,0.95,0.003,1000,20);
    agent.train();

//    ModelA2CGW model(4,64,128,64);
//    float a = 2.2;
//    torch::Tensor x = torch::tensor({a,a,a,a}).view({-1,1});
//    x = model.actorOutput(x);
//    cout<<x<<endl;
//    return a.exec();

}
