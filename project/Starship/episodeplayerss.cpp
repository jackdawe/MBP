#include "episodeplayerss.h"

EpisodePlayerSS::EpisodePlayerSS()
{

}

EpisodePlayerSS::EpisodePlayerSS(string filename):
  wpColors(QList<QColor>({Qt::red,Qt::green,Qt::yellow,Qt::cyan,Qt::black}))
{
  map.load(filename);
  initMap();
}

void EpisodePlayerSS::initMap()
{
  connect(&playClock,SIGNAL(timeout()),this,SLOT(update()));
  connect(&signalClock,SIGNAL(timeout()),this,SLOT(signalOff()));
  ship.setWidth(SHIP_WIDTH);
  ship.setHeight(SHIP_HEIGHT);
  ssView.setScene(&ssScene);
  ssScene.setSceneRect(0,0,map.getSize(),map.getSize());
  ssView.setFixedSize(map.getSize(),map.getSize());
  for (unsigned int i=0;i<map.getPlanets().size();i++)
    {
      planetShapes.push_back(new QGraphicsEllipseItem(0,0,map.getPlanets()[i].getRadius()*2,map.getPlanets()[i].getRadius()*2));
      planetShapes.last()->setBrush(QBrush(Qt::blue));
      planetShapes.last()->setPos(map.getPlanets()[i].getCentre().x-map.getPlanets()[i].getRadius(), map.getPlanets()[i].getCentre().y-map.getPlanets()[i].getRadius());
      ssScene.addItem(planetShapes.last());        
    }
  for (unsigned int i=0;i<map.getWaypoints().size();i++)
    {
      waypointShapes.push_back(new QGraphicsEllipseItem(0,0,map.getWaypoints()[i].getRadius()*2,map.getWaypoints()[i].getRadius()*2));
      waypointShapes.last()->setBrush(wpColors[i]);
      waypointShapes.last()->setPos(map.getWaypoints()[i].getCentre().x-map.getWaypoints()[i].getRadius(), map.getWaypoints()[i].getCentre().y-map.getWaypoints()[i].getRadius());
      ssScene.addItem(waypointShapes.last());
    }
  QPolygonF shipTriangle;
  int w = ship.getWidth();
  int h = ship.getHeight();  
  shipTriangle.append(QPointF(0,(2*h/3.)));
  shipTriangle.append(QPointF(-w/2.,-h/3.));
  shipTriangle.append(QPointF(w/2.,-h/3));
  shipShape = new QGraphicsPolygonItem(shipTriangle);
  shipShape->setBrush(QBrush(Qt::magenta));
  ssScene.addItem(shipShape);

  QPolygonF thrustTriangle;      
  thrustTriangle.append(QPointF(-w/4,-h/9));
  thrustTriangle.append(QPointF(w/4,-h/9));
  thrustTriangle.append(QPointF(0,2*h/9));
  thrustShape = new QGraphicsPolygonItem(thrustTriangle);
  thrustShape->setBrush(QBrush(Qt::darkYellow));      
  ssScene.addItem(thrustShape);
  
  signalShape =new QGraphicsEllipseItem(0,0,ship.getWidth()*2/3.,ship.getWidth()*2/3.);
  signalShape->setBrush(QBrush(Qt::magenta));
  ssScene.addItem(signalShape);  
  shipShape->hide();
  signalShape->hide();
  thrustShape->hide();
}

void EpisodePlayerSS::showMap()
{
    ssView.show();
}

void EpisodePlayerSS::playEpisode(vector<vector<float>> actionSequence, vector<vector<float>> stateSequence, float maxThrust)
{
  this->maxThrust = maxThrust;
  this->actionSequence = actionSequence;
  this->stateSequence = stateSequence;
  shipShape->show();
  thrustShape->show();
  signalShape->show(); 
  shipShape->setPos(stateSequence[0][0],stateSequence[0][1]);
  thrustShape->setPos(stateSequence[0][0],stateSequence[0][1]-ship.getWidth()/3);
  signalShape->setPos(stateSequence[0][0]-ship.getWidth()*9/24.,stateSequence[0][1]-ship.getWidth()*7/12.);
  showMap();
  playClock.start(TIME_STEP);
  stepCount=0;
}

void EpisodePlayerSS::signalOff()
{
  signalClock.stop();
  signalShape->setBrush(QBrush(Qt::magenta));
}

void EpisodePlayerSS::update()
{
  if (stepCount == stateSequence.size()-1 || stepCount == 3000)
    {
      playClock.stop();
    }
  else
    {
      Vect2d thrust(actionSequence[stepCount][1],actionSequence[stepCount][2]);
      float thrustPow = thrust.norm();
      float thrustO = thrust.angle();
      int signalColor = actionSequence[stepCount][0];
      if (thrustPow == 0)
	{
	  thrustPow=maxThrust/100.;
	}      
      if (signalColor != map.getWaypoints().size())
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

      int w = ship.getWidth();
      int h = ship.getHeight();
      ssScene.removeItem(thrustShape);
      QPolygonF thrustTriangle;
      thrustTriangle.append(QPointF(-3*w/8,2*h/9));
      thrustTriangle.append(QPointF(3*w/8,2*h/9));
      thrustTriangle.append(QPointF(0,-(thrustPow/maxThrust)*(6*h/9)));
      thrustTriangle.translate(0,-10*h/9);
      thrustShape = new QGraphicsPolygonItem(thrustTriangle);
      thrustShape->setBrush(QBrush(Qt::yellow));
      ssScene.addItem(thrustShape);      
      
      signalShape->setPos(stateSequence[stepCount][0]-ship.getWidth()*9/24.,stateSequence[stepCount][1]-ship.getWidth()*7/12.);
      shipShape->setRotation(Vect2d(cos(thrustO),sin(thrustO)).dilate(thrustPow).angle()*180/M_PI+90);
      thrustShape->setPos(stateSequence[stepCount][0],stateSequence[stepCount][1]+h/2);      
      thrustShape->setTransformOriginPoint(0,-(h/3.)-(thrustPow/maxThrust)*(2*h/9));
      thrustShape->setRotation(Vect2d(cos(thrustO),sin(thrustO)).dilate(thrustPow).angle()*180/M_PI+90);
      signalShape->setTransformOriginPoint(9*ship.getWidth()/24.,7*ship.getWidth()/12.);
      stepCount++;	
    }    
}

EpisodePlayerSS::~EpisodePlayerSS()
{
  delete shipShape;
  delete signalShape;
  for (unsigned int i=0;i<map.getPlanets().size();i++)
    {
      delete planetShapes[i];
    }
  for (unsigned int i=0;i<map.getWaypoints().size();i++)
    {
      delete waypointShapes[i];
    }
}
