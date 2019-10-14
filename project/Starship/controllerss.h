#ifndef CONTROLLERSS_H
#define CONTROLLERSS_H
#include "controller.h"
#include "mapss.h"

#define GRAVITY 0.01
#define DAMPING 0.1
#define PLANET_DENSITY 0.003
#define SHIP_MASS 2
#define SHIP_MAX_THRUST 2
#define STEP_SIZE 0.5
#define RIGHT_SIGNAL_ON_WAYPOINT_REWARD 1
#define WRONG_SIGNAL_ON_WAYPOINT_REWARD -1
#define SIGNAL_OFF_WAYPOINT_REWARD -0.1
#define CRASH_REWARD -1

class ControllerSS: public Controller
{
public:

    ControllerSS();
    ControllerSS(string mapTag);
    ControllerSS(string mapTag, Ship s);

    float transition();
    bool isTerminal(State s);
    void generateStates();
    void updateStates();
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();


    vector<Planet> planets;
    vector<Waypoint> waypoints;
    Ship ship;

};

#endif // CONTROLLERSS_H
