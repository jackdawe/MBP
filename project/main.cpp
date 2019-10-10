#include <QApplication>
#include "GridWorld/stategw.h"
#include "actionspace.h"
#include "GridWorld/controllergw.h"
#include "GridWorld/episodeplayergw.h"
#include "agenttrainer.h"
#include "Agents/qlearning.h"

int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);
    ControllerGW controller("2_8");
    QLearning<ControllerGW> agent(controller,0.05,0.9);
    AgentTrainer<QLearning<ControllerGW>> trainer;
    trainer.train(&agent,1000,1);
//    return a.exec();
}
