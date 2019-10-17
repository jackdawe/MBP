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

//    AgentTrainer<QLearning<ControllerGW>> trainer;
//    vector<vector<int>> r;
//    for (int i=0;i<16;i++)
//    {
//        vector<int> rl;
//        for (int j=0;j<16;j++)
//        {
//            ControllerGW controller(mapTag,i,j);
//            QLearning<ControllerGW> agent(controller,0.25,0.9);
//            agent.loadPolicy("E025G090_1900");
//            agent.epsilon = 0;
//            trainer.train(&agent,1,0,0);
//            rl.push_back((int)agent.getController().getRewardHistory().front());
//        }
//        r.push_back(rl);
//    }
//    EpisodePlayerGW ep(mapTag);
//    ep.showScores(r);
//    return a.exec();
}
