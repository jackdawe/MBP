#ifndef EPISODEPLAYERSS_H
#define EPISODEPLAYERSS_H
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QTimer>
#include "mapss.h"

#define TIME_STEP 50 //in ms

class EpisodePlayerSS: public QWidget
{
    Q_OBJECT
public:
    EpisodePlayerSS();
    EpisodePlayerSS(string mapTag);
    EpisodePlayerSS(string mapTag, vector<vector<float>> actionSequence,vector<vector<float>> stateSequence,
                    vector<float> parameters);
    void initMap();
    void showMap();
    void playEpisode();
    ~EpisodePlayerSS();

public slots:
    void update();

private:
    vector<vector<float>> actionSequence;
    vector<vector<float>> stateSequence;
    vector<float> parameters;
    MapSS map;
    Ship ship;
    QTimer playClock;
    QTimer signalClock;
    unsigned int stepCount;
    QGraphicsScene ssScene;
    QGraphicsView ssView;
    QList<QGraphicsEllipseItem*> planetShapes;
    QList<QGraphicsEllipseItem*> waypointShapes;
    QGraphicsPolygonItem *shipShape;
    QGraphicsEllipseItem *signalShape;
    QList<QColor> wpColors;
};

#endif // EPISODEPLAYERSS_H
