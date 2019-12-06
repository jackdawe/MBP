#ifndef SPACEWORLD_H
#define SPACEWORLD_H
#include "world.h"
#include "mapss.h"

#define GRAVITY 0.01
#define DAMPING 0.1
#define PLANET_DENSITY 1
#define SHIP_MASS 2
#define SHIP_MAX_THRUST 0.01
#define STEP_SIZE 0.5
#define RIGHT_SIGNAL_ON_WAYPOINT_REWARD 1
#define CRASH_REWARD -1
#define WRONG_SIGNAL_ON_WAYPOINT_REWARD -1
#define SIGNAL_OFF_WAYPOINT_REWARD -0.1


class SpaceWorld: public World
{
public:

    SpaceWorld();
    SpaceWorld(string filename);
    SpaceWorld(string filename, Ship s);

    float transition();
    bool isTerminal(State s);
    void generateVectorStates();
    void reset();

    int size;
    vector<Planet> planets;
    vector<Waypoint> waypoints;
    Ship ship;
};

#endif // SPACEWORLD_H
