#include <QApplication>
#include "GridWorld/gridworld.h"
#include "Agents/qlearning.h"
#include "Agents/A2C/actorcritic.h"
#include "GridWorld/episodeplayergw.h"
int main(int argc, char *argv[])
{
//    QApplication a(argc, argv);

    GridWorld gw("2_16",1,14);
    gw.generateVectorStates();
    cv::Mat rgbState = gw.toRGB(gw.getCurrentState());
    cv::Mat dst;
    cv::resize(rgbState,dst,cv::Size(800,800));
    string WindowName = "hey";
    cv::namedWindow(WindowName);
    cv::imshow(WindowName,dst);
    cv::waitKey(0);
    cv::destroyWindow(WindowName);
//    a.exec();
}


