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
public:
    EpisodePlayerSS();
    EpisodePlayerSS(string mapTag);
    EpisodePlayerSS(string mapTag, vector<vector<float>> sequence);
    void initMap();
    void showMap();
    void playEpisode();
    ~EpisodePlayerSS();

public slots:
    void update();

private:
    vector<vector<float>> sequence;
    MapSS map;
    QTimer playClock;
    unsigned int stepCount;
    QGraphicsScene ssScene;
    QGraphicsView ssView;
    QList<QGraphicsEllipseItem*> planetShapes;
    QList<QGraphicsEllipseItem*> waypointShapes;
    QGraphicsPolygonItem *shipShape;
    QList<QColor> wpColors;
};

#endif // EPISODEPLAYERSS_H
