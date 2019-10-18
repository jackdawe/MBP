#include <QApplication>
#include "GridWorld/stategw.h"
#include "actionspace.h"
#include "GridWorld/controllergw.h"
#include "agenttrainer.h"
#include "Agents/qlearning.h"
#include "Starship/episodeplayerss.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    string mapTag = "2_16";
//    ControllerGW controller(mapTag);
//    QLearning<ControllerGW> agent(controller,0.25,0.9);
//    AgentTrainer<QLearning<ControllerGW>> trainer;
//    trainer.train(&agent,100000,1);
//    agent.savePolicy();
//    agent.saveRewardHistory();

//    trainer.train(&agent,1,0,1);
//    agent.loadEspisode(agent.getNameTag());
//    EpisodePlayerGW ep(mapTag,agent.getController().getStateSequence());
//    ep.playEpisode();

//    return a.exec();
}
