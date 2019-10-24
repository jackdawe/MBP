#include <QApplication>
#include "GridWorld/gridworld.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"
int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);


    GridWorld gw("2_8");
    ConvNetGW net(8,16,32,8*8*2);
    gw.generateVectorStates();
    torch::Tensor s = gw.toRGBTensor(gw.getCurrentState());
    torch::Tensor x = net.actorOutput(s);
    cout << x << endl;
//    a.exec();
}


