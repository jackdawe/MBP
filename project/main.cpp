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
    QLearning<ControllerGW> agent(controller,0.25,0.9);
    AgentTrainer<QLearning<ControllerGW>> trainer;
    trainer.train(&agent,100,1);
//    agent.saveRewardHistory();
//    agent.savePolicy();

    QLearning<ControllerGW> agent2;
    agent2.loadPolicy("E025G090_140");

//    trainer.train(&agent,1,1,1);
//    trainer.loadSequence(0);
//    EpisodePlayerGW ep(mapTag,trainer.getStateSequence());
//    ep.playEpisode();
    return a.exec();
}
