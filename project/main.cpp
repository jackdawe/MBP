#include <QCoreApplication>
#include "GridWorld/stategw.h"
#include "actionspace.h"
#include "GridWorld/controllergw.h"

int main(int argc, char *argv[])
{
//    QCoreApplication a(argc, argv);
//    return a.exec();
    ControllerGW c("1_8");
    bool t = c.isTerminal(c.getCurrentState());
}
