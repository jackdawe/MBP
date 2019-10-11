#include "episodeplayerss.h"

EpisodePlayerSS::EpisodePlayerSS()
{

}

EpisodePlayerSS::EpisodePlayerSS(string mapTag):
    wpColors(QList<QColor>({Qt::red,Qt::green,Qt::yellow,Qt::cyan,Qt::black}))
{
    map.load(mapTag);
    initMap();
}

EpisodePlayerSS::EpisodePlayerSS(string mapTag, vector<vector<float> > sequence):
    wpColors(QList<QColor>({Qt::red,Qt::green,Qt::yellow,Qt::cyan,Qt::black}))
{
    map.load(mapTag);
    initMap();
    connect(&playClock,SIGNAL(timeout()),this,SLOT(update()));
}

void EpisodePlayerSS::initMap()
{
    ssView.setScene(&ssScene);
    ssScene.setSceneRect(0,0,map.getSize(),map.getSize());
    ssView.setFixedSize(map.getSize(),map.getSize());
    for (unsigned int i=0;i<map.getPlanets().size();i++)
    {
        planetShapes.push_back(new QGraphicsEllipseItem(0,0,map.getPlanets()[i].getRadius(),map.getPlanets()[i].getRadius()));
        planetShapes.last()->setBrush(QBrush(Qt::blue));
        planetShapes.last()->setPos(map.getPlanets()[i].getCentre().getX(),map.getPlanets()[i].getCentre().getY());
        ssScene.addItem(planetShapes.last());
    }
    for (unsigned int i=0;i<map.getWaypoints().size();i++)
    {
        waypointShapes.push_back(new QGraphicsEllipseItem(0,0,map.getWaypoints()[i].getRadius(),map.getWaypoints()[i].getRadius()));
        waypointShapes.last()->setBrush(wpColors[i]);
        waypointShapes.last()->setPos(map.getWaypoints()[i].getCentre().getX(),map.getWaypoints()[i].getCentre().getY());
        ssScene.addItem(waypointShapes.last());
    }
    QPolygonF shipTriangle;
    shipTriangle.append(QPointF(0,(2*map.getShip().getHeight()/3)));
    shipTriangle.append(QPoint(-map.getShip().getWidth()/2,-map.getShip().getHeight()/3));
    shipTriangle.append(QPoint(map.getShip().getWidth()/2,-map.getShip().getHeight()/3));
    shipShape = new QGraphicsPolygonItem(shipTriangle);
    shipShape->setBrush(QBrush(Qt::magenta));
    shipShape->setPos(map.getShip().getP().getX(),map.getShip().getP().getY());
    ssScene.addItem(shipShape);
}

void EpisodePlayerSS::showMap()
{
    ssView.show();
}

EpisodePlayerSS::~EpisodePlayerSS()
{
    delete shipShape;
    for (unsigned int i=0;i<map.getPlanets().size();i++)
    {
        delete planetShapes[i];
    }
    for (unsigned int i=0;i<map.getWaypoints().size();i++)
    {
        delete waypointShapes[i];
    }
}
