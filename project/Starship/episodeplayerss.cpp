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

EpisodePlayerSS::EpisodePlayerSS(string mapTag, vector<vector<float>> actionSequence,vector<vector<float>> stateSequence,
                                 vector<float> parameters):
    wpColors(QList<QColor>({Qt::red,Qt::green,Qt::yellow,Qt::cyan,Qt::black})), actionSequence(actionSequence),
    stateSequence(stateSequence), parameters(parameters), stepCount(0)
{
    map.load(mapTag);    
    initMap();
    connect(&playClock,SIGNAL(timeout()),this,SLOT(update()));
}

void EpisodePlayerSS::initMap()
{
    ship = map.getShip();
    ssView.setScene(&ssScene);
    ssScene.setSceneRect(0,0,map.getSize(),map.getSize());
    ssView.setFixedSize(map.getSize(),map.getSize());
    for (unsigned int i=0;i<map.getPlanets().size();i++)
    {
        planetShapes.push_back(new QGraphicsEllipseItem(0,0,map.getPlanets()[i].getRadius()*2,map.getPlanets()[i].getRadius()*2));
        planetShapes.last()->setBrush(QBrush(Qt::blue));
        planetShapes.last()->setPos(map.getPlanets()[i].getCentre().getX()-map.getPlanets()[i].getRadius(),
                                    map.getPlanets()[i].getCentre().getY()-map.getPlanets()[i].getRadius());
        ssScene.addItem(planetShapes.last());        
    }
    for (unsigned int i=0;i<map.getWaypoints().size();i++)
    {
        waypointShapes.push_back(new QGraphicsEllipseItem(0,0,map.getWaypoints()[i].getRadius(),map.getWaypoints()[i].getRadius()));
        waypointShapes.last()->setBrush(wpColors[i]);
        waypointShapes.last()->setPos(map.getWaypoints()[i].getCentre().getX()-map.getWaypoints()[i].getRadius(),
                                      map.getWaypoints()[i].getCentre().getY()-map.getWaypoints()[i].getRadius());
        ssScene.addItem(waypointShapes.last());
    }
    QPolygonF shipTriangle;
    shipTriangle.append(QPointF(0,(2*ship.getHeight()/3)));
    shipTriangle.append(QPoint(-ship.getWidth()/2,-ship.getHeight()/3));
    shipTriangle.append(QPoint(ship.getWidth()/2,-ship.getHeight()/3));
    shipShape = new QGraphicsPolygonItem(shipTriangle);
    shipShape->setBrush(QBrush(Qt::magenta));
    shipShape->setPos(ship.getP().getX(),ship.getP().getY());
    ssScene.addItem(shipShape);

    signalShape = new QGraphicsEllipseItem(0,0,ship.getWidth()*2/3,ship.getWidth()*2/3);
    signalShape->setBrush(QBrush(Qt::magenta));
    signalShape->setPos(ship.getP().getX()-ship.getWidth()*9/24,ship.getP().getY()-ship.getWidth()*7/12);
    ssScene.addItem(signalShape);

}

void EpisodePlayerSS::showMap()
{
    ssView.show();
}

void EpisodePlayerSS::playEpisode()
{
    showMap();
    playClock.start(50);
}

void EpisodePlayerSS::update()
{
    if (stepCount == stateSequence.size()-1)
    {
        playClock.stop();
    }
    else
    {
        stepCount++;
        float thrustPow = actionSequence[stepCount][1];
        float thrustO = actionSequence[stepCount][2];
        int signalColor = actionSequence[stepCount][0];
        if (signalColor != wpColors.size())
        {
            QColor a = wpColors[signalColor];
            signalShape->setBrush(QBrush(wpColors[signalColor]));
            signalClock.start(100);
        }
        else
        {
            signalShape->setBrush(QBrush(Qt::magenta));
        }
        shipShape->setPos(stateSequence[stepCount][0],stateSequence[stepCount][1]);
        signalShape->setPos(stateSequence[stepCount][0]-ship.getWidth()*9/24,stateSequence[stepCount][1]-ship.getWidth()*7/12);
        shipShape->setRotation(Vect2d(cos(thrustO),sin(thrustO)).dilate(thrustPow).angle()*180/M_PI-90);
        signalShape->setTransformOriginPoint(9*ship.getWidth()/24,7*ship.getWidth()/12);
    }
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
