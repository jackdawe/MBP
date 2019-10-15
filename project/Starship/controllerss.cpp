#include "controllerss.h"

ControllerSS::ControllerSS()
{

}

ControllerSS::ControllerSS(string mapTag)
{
    paramLabels = {"Gavity", "Damping","Planet Density", "Ship Mass", "Ship Maximum Thrust", "Step Size", "Win Reward",
                   "Lose Reward", "Wrong signal on Waypoint Reward", "Signal Off Waypoint Reward"};
    paramValues = {GRAVITY,DAMPING,PLANET_DENSITY,SHIP_MASS,SHIP_MAX_THRUST,STEP_SIZE,RIGHT_SIGNAL_ON_WAYPOINT_REWARD,
                   CRASH_REWARD, WRONG_SIGNAL_ON_WAYPOINT_REWARD,SIGNAL_OFF_WAYPOINT_REWARD};
    MapSS map;
    map.load(mapTag);
    size = map.getSize();
    planets = map.getPlanets();
    for (unsigned int i=0;i<planets.size();i++)
    {
        planets[i].setMass(4*PLANET_DENSITY*pow(planets[i].getRadius(),3)*M_PI/3);
    }
    waypoints = map.getWaypoints();
    ship = map.getShip();
    vector<DiscreteAction> dactions = {DiscreteAction(waypoints.size()+1)};
    vector<ContinuousAction> cactions {ContinuousAction(0,SHIP_MAX_THRUST), ContinuousAction(0,2*M_PI)};
    actions = ActionSpace(dactions,cactions);
    takenAction = vector<float>(3,0);
}

ControllerSS::ControllerSS(string mapTag, Ship s)
{

}

float ControllerSS::transition()
{

    previousState.update(0,ship.getP().getX()),previousState.update(1,ship.getP().getY()),
            previousState.update(2,ship.getV().getX()), previousState.update(3,ship.getV().getY());

    float thrustPow = takenAction[1];
    float thrustOri = takenAction[2];
    int signal = (int)takenAction[0];

    if (!isTerminal(currentState))
    {
        ship.setThrust(Vect2d(cos(thrustOri),sin(thrustOri)).dilate(thrustPow));
        ship.setSignalColor(signal);
    }
    Vect2d gravForce(0,0);
    for (unsigned int i=0;i<planets.size();i++)
    {
        Planet p = planets[i];
        Vect2d vectPS = p.getCentre().sum(ship.getP().dilate(-1));
        gravForce = gravForce.sum(vectPS.dilate(GRAVITY*SHIP_MASS*p.getMass()/pow(vectPS.norm(),3)));
        ship.setA(Vect2d(gravForce.getX()-DAMPING*ship.getV().getX()-ship.getThrust().getX(),gravForce.getY()-DAMPING*ship.getV().getY()-ship.getThrust().getY()).dilate(1./SHIP_MASS));
        ship.setP(ship.getP().sum(ship.getV().dilate(STEP_SIZE)));
        ship.setV(ship.getV().sum(ship.getA().dilate(STEP_SIZE)));
    }
    for (unsigned int i=0;i<planets.size();i++)
    {
        if (ship.getP().distance(planets[i].getCentre()) < ship.getWidth()+planets[i].getRadius())
        {
            return CRASH_REWARD;
        }
    }

    currentState.update(0,ship.getP().getX()),currentState.update(1,ship.getP().getY()),
            currentState.update(2,ship.getV().getX()), currentState.update(3,ship.getV().getY());

    if (ship.getSignalColor() != actions.getDiscreteActions()[0].getSize())
    {
        for (unsigned int i=0;i<waypoints.size();i++)
        {
            if (ship.getP().distance(waypoints[i].getCentre()) < waypoints[i].getRadius())
            {
                if (ship.getSignalColor() == i)
                {
                    return RIGHT_SIGNAL_ON_WAYPOINT_REWARD;
                }
                else
                {
                    return WRONG_SIGNAL_ON_WAYPOINT_REWARD;
                }
            }
            return SIGNAL_OFF_WAYPOINT_REWARD;
        }
    }
    return 0;
}

bool ControllerSS::isTerminal(State s)
{
    for (unsigned int i=0;i<planets.size();i++)
    {
        if (ship.getP().distance(planets[i].getCentre()) < ship.getWidth()+planets[i].getRadius())
        {
            return true;
        }
    }
    if (ship.getSignalColor() != actions.getDiscreteActions()[0].getSize())
    {
        for (unsigned int i=0;i<waypoints.size();i++)
        {
            if (ship.getP().distance(waypoints[i].getCentre()) < waypoints[i].getRadius())
            {
                if (ship.getSignalColor() == i)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

void ControllerSS::generateStates()
{
    vector<float> stateVector = {ship.getP().getX(),ship.getP().getY(),ship.getV().getX(),
                                 ship.getV().getY(),ship.getWidth()};
    for (unsigned int i=0;i<waypoints.size();i++)
    {
        stateVector.push_back(waypoints[i].getCentre().getX());
        stateVector.push_back(waypoints[i].getCentre().getY());
        stateVector.push_back(waypoints[i].getRadius());
    }
    for (unsigned int i=0;i<planets.size();i++)
    {
        stateVector.push_back(planets[i].getCentre().getX());
        stateVector.push_back(planets[i].getCentre().getY());
        stateVector.push_back(planets[i].getRadius());
    }
    currentState.setStateVector(stateVector);
    previousState.setStateVector(vector<float>(stateVector.size(),0));
}

void ControllerSS::reset()
{
    default_random_engine generator(random_device{}());
    uniform_int_distribution<int> dist(ship.getHeight(),size-ship.getHeight());
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
    ship.setA(Vect2d(0,0));
    ship.setV(Vect2d(0,0));
    ship.setThrust(Vect2d(0,0));
    ship.setSignalColor(waypoints.size());
}

