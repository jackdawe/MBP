#ifndef CONTROLLERSS_H
#define CONTROLLERSS_H
#include "controller.h"
#include "planet.h"
#include "waypoint.h"
#include "ship.h"

class ControllerSS: public Controller
{
public:

    ControllerSS();

    float transition();
    bool isTerminal();
    void updateStateVector();
    int stateId(State s);
    void reset();
    vector<int> accessibleStates(State s);
    int spaceStateSize();

    vector<Planet> planets;
    vector<Waypoint> Waypoint;


};

#endif // CONTROLLERSS_H
