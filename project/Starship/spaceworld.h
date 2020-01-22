#ifndef SPACEWORLD_H
#define SPACEWORLD_H
#include "world.h"
#include "ship.h"
#include "mapss.h"

#define SHIP_WIDTH 20
#define SHIP_HEIGHT 40
#define GRAVITY 0.05
#define DAMPING 0.1
#define SHIP_MASS 1
#define SHIP_MAX_THRUST 1
#define STEP_SIZE 4
#define RIGHT_SIGNAL_ON_WAYPOINT_REWARD 1
#define CRASH_REWARD -1
#define WRONG_SIGNAL_ON_WAYPOINT_REWARD -1
#define SIGNAL_OFF_WAYPOINT_REWARD -0.1
#define EPISODE_LENGTH 80

class SpaceWorld: public World
{
public:

    SpaceWorld();
    SpaceWorld(string filename);
    SpaceWorld(string filename, Ship s);
    SpaceWorld(string pathToDir, int mapPoolSize);
    
    float transition();
    bool isTerminal(State s);
    void generateVectorStates();
    void reset();

    void placeShip();
    bool isCrashed();

    int getSvSize();
    int getSize();    
    vector<Waypoint> getWaypoints();
    Ship getShip();
    void repositionShip(Vect2d p);
    int epCount;
    
 private:
    void init();

    int size;
    int svSize;
    MapSS map;
    bool randomStart;
    string mapPoolPath;
    int mapPoolSize;
    vector<Planet> planets;
    vector<Waypoint> waypoints;
    Ship initShip;
    Ship ship;
};

#endif // SPACEWORLD_H
