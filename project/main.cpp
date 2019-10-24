#include <QApplication>
#include "GridWorld/controllergw.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    a.exec();
}


