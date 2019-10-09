#ifndef EPISODEPLAYERGW_H
#define EPISODEPLAYERGW_H
#include "mapgw.h"
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <QTimer>

#define TIME_STEP 500 //in ms
#define SQUARE_SIZE 50 //in pixels

class EpisodePlayerGW: public QWidget
{
    Q_OBJECT
public:
    EpisodePlayerGW();
    EpisodePlayerGW(MapGW map);
    EpisodePlayerGW(MapGW map,vector<vector<double>> sequence);
    void initMap();
    void showMap();
    void playEpisode();
    MapGW getMap();
    vector<vector<double>> getSequence();
    void setMap(MapGW map);
    void setSequence(vector<vector<double>> sequence);
    ~EpisodePlayerGW();

public slots:
    void update();

private:
    vector<vector<double>> sequence;
    MapGW map;
    QTimer playClock;
    unsigned int stepCount;
    QGraphicsScene gwScene;
    QGraphicsView gwView;
    QList<QGraphicsRectItem*> obstacleShapes;
    QGraphicsRectItem *agentShape;
    QGraphicsRectItem *startShape;
    QGraphicsRectItem *arrivalShape;
};

#endif // EPISODEPLAYERGW_H
