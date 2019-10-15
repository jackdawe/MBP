#include <QApplication>
#include "GridWorld/stategw.h"
#include "actionspace.h"
#include "GridWorld/controllergw.h"
#include "GridWorld/episodeplayergw.h"
#include "agenttrainer.h"
#include "Agents/qlearning.h"
#include "Starship/episodeplayerss.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    string mapTag = "2_8";
    ControllerGW controller(mapTag);
    QLearning<ControllerGW> agent(controller,0.05,0.9);
    AgentTrainer<QLearning<ControllerGW>> trainer;
    trainer.train(&agent,10000,1,0);

    trainer.train(&agent,1,1,1);
    trainer.loadSequence(0);
    EpisodePlayerGW ep(mapTag,trainer.getStateSequence());
    ep.playEpisode();


//    MapSS map(1000,"1");
//    map.generate(1);
//    map.save();
//    EpisodePlayerSS ep("1");
//    ep.showMap();
    return a.exec();
}
