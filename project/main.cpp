#include <QApplication>
#include "GridWorld/stategw.h"
#include "actionspace.h"
#include "GridWorld/controllergw.h"
#include "GridWorld/episodeplayergw.h"

int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);

    ControllerGW c("1_8");
    bool t = c.isTerminal(c.getCurrentState());
//    return a.exec();
}
