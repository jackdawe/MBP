#include <QApplication>
#include "GridWorld/stategw.h"
#include "actionspace.h"
#include "GridWorld/controllergw.h"
#include "agenttrainer.h"
#include "Agents/qlearning.h"
#include "Starship/episodeplayerss.h"
#include "Agents/actorcritic.h"

int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);

    string mapTag = "2_8";
    ControllerGW c(mapTag);
    ModelA2CGW model(c.getSize()*c.getSize()+4,64,128,64);
    ActorCritic<ControllerGW,ModelA2CGW> agent(model,0.95,0.003,10,20);
    agent.train();

//    return a.exec();

}
