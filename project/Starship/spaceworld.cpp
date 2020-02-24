#include "spaceworld.h"

SpaceWorld::SpaceWorld(){}

SpaceWorld::SpaceWorld(string filename):
  randomStart(true), mapPoolSize(-1)
{
  map.load(filename);
  init();
}

SpaceWorld::SpaceWorld(vector<float> initStateVector, int nWaypoints):
  randomStart(false), mapPoolSize(-1)
{
  ship.setA(Vect2d(0,0));
  ship.setV(Vect2d(initStateVector[2],initStateVector[3]));
  ship.setP(Vect2d(initStateVector[0],initStateVector[1]));
  map = MapSS(initStateVector, nWaypoints);
  init();
  initShip = Ship(ship);
}

SpaceWorld::SpaceWorld(string filename, Ship s):
  ship(s), randomStart(false), mapPoolSize(-1)
{
  map.load(filename);
  init();
  initShip = Ship(ship);
}

SpaceWorld::SpaceWorld(string pathToDir, int mapPoolSize):
  mapPoolPath(pathToDir), randomStart(true), mapPoolSize(mapPoolSize)
{
  map.load(mapPoolPath+"map0");
  init();
}

void SpaceWorld::init()
{
  epCount=0;
  vector<DiscreteAction> dactions = {DiscreteAction(map.getWaypoints().size()+1)};
  vector<ContinuousAction> cactions = {ContinuousAction(-SHIP_MAX_THRUST,SHIP_MAX_THRUST), ContinuousAction(-SHIP_MAX_THRUST,SHIP_MAX_THRUST)};
  actions = ActionSpace(dactions,cactions);
  takenAction = vector<float>(3,0);
  svSize = 4+3*(map.getPlanets().size()+map.getWaypoints().size());
  currentState.setStateVector(vector<float>(svSize,0));
  ship.setWidth(SHIP_WIDTH);
  ship.setHeight(SHIP_HEIGHT);
  reset();
}

float SpaceWorld::transition(vector<float> action)
{
  previousState.update(0,ship.getP().x),previousState.update(1,ship.getP().y);
  previousState.update(2,ship.getV().x), previousState.update(3,ship.getV().y);
  takenAction = action;
  int signal = (int)action[0];
  Vect2d thrust(action[1],action[2]);  
  if(thrust.norm() > SHIP_MAX_THRUST)
    {
      thrust = thrust.dilate(SHIP_MAX_THRUST/thrust.norm());
    }
  ship.setThrust(thrust);
  ship.setSignalColor(signal);
  
  float r = 0;  
  if (!isTerminal(currentState))
    {
      //Transition function

      if (!isCrashed())
	{
	  Vect2d gravForce(0,0);
	  for (unsigned int i=0;i<planets.size();i++)
	    {
	      Planet p = planets[i];
	      Vect2d vectPS = p.getCentre().sum(ship.getP().dilate(-1));
	      gravForce = gravForce.sum(vectPS.dilate(GRAVITY*SHIP_MASS*p.getMass()/pow(vectPS.norm(),2)));
	      ship.setA(Vect2d(gravForce.x-DAMPING*ship.getV().x-ship.getThrust().x,gravForce.y-DAMPING*ship.getV().y-ship.getThrust().y).dilate(1./SHIP_MASS));
	      Vect2d newP = ship.getP().sum(ship.getV().dilate(STEP_SIZE));
	      if (newP.x>SIZE)
		{
		  newP.x-=SIZE;
		}
	      if (newP.x<0)
		{
		  newP.x+=SIZE;
		}
	      if (newP.y>SIZE)
		{
		  newP.y-=SIZE;
		}
	      if (newP.y<0)
		{
		  newP.y+=SIZE;
		}
	      ship.setP(newP);
	      if (!isCrashed())
		{
		  ship.setV(ship.getV().sum(ship.getA().dilate(STEP_SIZE)));
		}
	    }
	}
      currentState.update(0,ship.getP().x),currentState.update(1,ship.getP().y);
      currentState.update(2,ship.getV().x), currentState.update(3,ship.getV().y);	  
      
      //Reward function
      
      if (isCrashed())
	{
	  r = CRASH_REWARD;
	}
      else
	{
	  /*
	  if (ship.getSignalColor() != actions.getDiscreteActions()[0].getSize()-1)
	    {
	      for (unsigned int i=0;i<waypoints.size();i++)
		{
		  if (ship.getP().distance(waypoints[i].getCentre()) < waypoints[i].getRadius())
		    {
		      if (ship.getSignalColor() == i)
			{
			  //			  cout<<"yes"<<endl;
			  r = RIGHT_SIGNAL_ON_WAYPOINT_REWARD;
			  break;
			}
		      else
			{
			  r = WRONG_SIGNAL_ON_WAYPOINT_REWARD;
			  break;
			}
		    }
		  else
		    {
		      r = SIGNAL_OFF_WAYPOINT_REWARD;	  		      
		    }
		}
	    }
	  */
	  ///*
	  for (unsigned int i=0;i<waypoints.size();i++)
	    {
	      if (ship.getP().distance(waypoints[i].getCentre()) < waypoints[i].getRadius())
		{
		  r = RIGHT_SIGNAL_ON_WAYPOINT_REWARD;
		}
	    }	  
	  //*/
	}
    }  
  actionSequence.push_back({signal,thrust.x,thrust.y});
  stateSequence.push_back(currentState.getStateVector());
  rewardHistory.back()+=r;
  epCount++;
  return r;
}

bool SpaceWorld::isTerminal(State s)
{
  return epCount>=EPISODE_LENGTH;
}

void SpaceWorld::generateVectorStates()
{
  currentState.update(0,ship.getP().x), currentState.update(1,ship.getP().y), currentState.update(2,ship.getV().x), currentState.update(3,ship.getV().y);
  for (unsigned int i=0;i<waypoints.size();i++)
    {
      currentState.update(3*i+4,waypoints[i].getCentre().x);
      currentState.update(3*i+5,waypoints[i].getCentre().y);
      currentState.update(3*i+6,waypoints[i].getRadius());
    }
  for (unsigned int i=0;i<planets.size();i++)
    {
      currentState.update(3*i+3*waypoints.size()+4,planets[i].getCentre().x);
      currentState.update(3*i+3*waypoints.size()+5,planets[i].getCentre().y);
      currentState.update(3*i+3*waypoints.size()+6,planets[i].getRadius());
    }
  previousState = currentState;
}

void SpaceWorld::reset()
{
  epCount=0;
  rewardHistory.push_back(0);
  if (mapPoolSize!=-1)
    {
      default_random_engine generator(std::random_device{}());
      uniform_int_distribution<int> dist(0,mapPoolSize-1);
      int mapId = dist(generator);
      map.load(mapPoolPath+"map"+to_string(mapId));
    }  
  if (planets.size() == 0 || mapPoolSize != -1)
    {
      planets = map.getPlanets();
      for (unsigned int i=0;i<planets.size();i++)
	{
	  planets[i].setMass(20*planets[i].getRadius());
	}
      waypoints = map.getWaypoints();
    }
  placeShip();
  generateVectorStates();
  stateSequence = {currentState.getStateVector()};  
}

void SpaceWorld::placeShip()
{
  ship.setSignalColor(waypoints.size());
  ship.setV(Vect2d(0,0));
  ship.setA(Vect2d(0,0));
  ship.setThrust(Vect2d(0,0));
  default_random_engine generator(std::random_device{}());
  uniform_int_distribution<int> dist(0,SIZE-ship.getHeight());
  if (randomStart)
    {
      bool invalidPosition = true;
      while (invalidPosition)
	{
	  invalidPosition = false;
	  Vect2d spawn(dist(generator),dist(generator));
	  ship.setP(spawn);
	  for (unsigned int i=0;i<planets.size();i++)
	    {
	      if (spawn.distance(planets[i].getCentre()) < 1.1 * (ship.getHeight()+planets[i].getRadius())) {
                invalidPosition = true;
                break;
	      }
        }
	  if (!invalidPosition)
	    {
	      for (unsigned int i=0;i<waypoints.size();i++)
		{
		  if (spawn.distance(waypoints[i].getCentre()) < 1.1 * (waypoints[i].getRadius() + ship.getHeight()))
		    {
		      invalidPosition = true;
		      break;
		    }
		}
	    }
	}
    }
  else
    {
      if (initShip.getWidth()>0)
	{
	  ship = initShip;
	}
    }
}

 bool SpaceWorld::isCrashed()
 {
   for (unsigned int i=0;i<planets.size();i++)
     {
       if (ship.getP().distance(planets[i].getCentre()) < ship.getWidth()+planets[i].getRadius())
	 {
	   return true;
	 }
     }
   return false;
 }

int SpaceWorld::getSvSize()
 {
   return svSize;
 }

vector<Waypoint> SpaceWorld::getWaypoints()
{
  return waypoints;
}

Ship SpaceWorld::getShip()
{
  return ship;
}

void SpaceWorld::repositionShip(Vect2d p)
{
  ship.setP(p);
  currentState.update(0,p.x), currentState.update(1,p.y);
}
