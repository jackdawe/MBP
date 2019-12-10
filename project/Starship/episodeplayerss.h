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
  EpisodePlayerSS(string filename);
  void initMap();
  void showMap();
  void playEpisode(vector<vector<float>> actionSequence,vector<vector<float>> stateSequence);
  ~EpisodePlayerSS();
  
  public slots:
    void update();
    void signalOff();
    
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
