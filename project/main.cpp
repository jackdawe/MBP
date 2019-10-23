#include <QApplication>
//#include "actionspace.h"
#include "GridWorld/controllergw.h"
//#include "agenttrainer.h"
#include "Agents/qlearning.h"
//#include "Starship/episodeplayerss.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    string mapTag = "2_16";
    ControllerGW c(mapTag);
    QLearning<ControllerGW> agent(c,5000,0.25,0.90);
    agent.train();
    agent.playOne();
    agent.loadEspisode("QL_98");
    EpisodePlayerGW ep(mapTag,agent.getController().getStateSequence());
    ep.playEpisode();


    a.exec();
}


