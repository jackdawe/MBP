#ifndef EPISODEPLAYERGW_H
#define EPISODEPLAYERGW_H
#include "mapgw.h"
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <QTimer>

#define TIME_STEP 300 //in ms
#define SQUARE_SIZE 50 //in pixels

class EpisodePlayerGW: public QWidget
{
    Q_OBJECT
public:
    EpisodePlayerGW();
    EpisodePlayerGW(string mapTag);
    EpisodePlayerGW(string mapTag,vector<vector<float>> sequence);
    void initMap();
    void showMap();
    void playEpisode();
    MapGW getMap();
    vector<vector<float>> getSequence();
    void setMap(MapGW map);
    void setSequence(vector<vector<float>> sequence);
    ~EpisodePlayerGW();

public slots:
    void update();

private:
    vector<vector<float>> sequence;
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
