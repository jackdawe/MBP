#include "episodeplayergw.h"

EpisodePlayerGW::EpisodePlayerGW()
{

}

EpisodePlayerGW::EpisodePlayerGW(string filename)
{
    map.load(filename);
    initMap();
    agentShape->hide();
}

void EpisodePlayerGW::initMap()
{
  connect(&playClock,SIGNAL(timeout()),this,SLOT(update()));
  gwView.setScene(&gwScene);
  gwScene.setSceneRect(0,0,map.getSize()*SQUARE_SIZE,map.getSize()*SQUARE_SIZE);
  gwView.setFixedSize(map.getSize()*SQUARE_SIZE,map.getSize()*SQUARE_SIZE);
  agentShape = new QGraphicsRectItem(0,0,4*SQUARE_SIZE/5.,4*SQUARE_SIZE/5.);
  startShape = new QGraphicsRectItem(0,0,SQUARE_SIZE,SQUARE_SIZE);
  arrivalShape = new QGraphicsRectItem(0,0,SQUARE_SIZE,SQUARE_SIZE);
  agentShape->setBrush(QBrush(Qt::magenta));
  agentShape->setZValue(1);
  gwScene.addItem(agentShape);
  startShape->setBrush(QBrush(Qt::yellow));
  gwScene.addItem(startShape);
  arrivalShape->setBrush(QBrush(Qt::green));
  gwScene.addItem(arrivalShape);
  for (int i=0;i<map.getSize();i++)
    {
      for (int j=0;j<map.getSize();j++)
        {
	  switch(map.getMap()[i][j])
            {
	    case 1:
	      obstacleShapes.append(new QGraphicsRectItem(0,0,SQUARE_SIZE,SQUARE_SIZE));
	      obstacleShapes.last()->setBrush(QBrush(Qt::red));
	      obstacleShapes.last()->setPos(SQUARE_SIZE*j,SQUARE_SIZE*i);
	      gwScene.addItem(obstacleShapes.last());
	      break;
	    case 2:
	      arrivalShape->setPos(SQUARE_SIZE*j,SQUARE_SIZE*i);
	      break;
            }
        }
    }
}

void EpisodePlayerGW::showMap()
{
    gwView.show();
}

void EpisodePlayerGW::displayOnGrid(vector<vector<string>> texts)
{
    for (unsigned int i=0;i<texts.size();i++)
    {
        for (unsigned int j=0;j<texts.size();j++)
        {
            QGraphicsTextItem *text= new QGraphicsTextItem;
            text->setPos(SQUARE_SIZE*(j),SQUARE_SIZE*(i+0.25));
            text->setPlainText(QString::fromStdString(texts[i][j]));
            gwScene.addItem(text);
        }
    }
    showMap();
}

void EpisodePlayerGW::playEpisode(vector<vector<float>> sequence)
{
  this->sequence = sequence;
  startShape->setPos(sequence[0][1]*SQUARE_SIZE,sequence[0][0]*SQUARE_SIZE);
  gwView.show();
  playClock.start(TIME_STEP);
  stepCount=0;
}

void EpisodePlayerGW::update()
{
  agentShape->show();
  if (stepCount == sequence.size()-1)
    {
      playClock.stop();
    }
  else
    {
      stepCount++;      
      agentShape->setPos((sequence[stepCount][1]+0.1)*SQUARE_SIZE,(sequence[stepCount][0]+0.1)*SQUARE_SIZE);
    }
}

MapGW EpisodePlayerGW::getMap()
{
    return map;
}

vector<vector<float>> EpisodePlayerGW::getSequence()
{
    return sequence;
}

void EpisodePlayerGW::setMap(MapGW map)
{
    this->map = map;
}

EpisodePlayerGW::~EpisodePlayerGW()
{
    delete agentShape;
    delete startShape;
    delete arrivalShape;
    for (int i=0;i<obstacleShapes.length();i++)
    {
        delete obstacleShapes.at(i);
    }
}
