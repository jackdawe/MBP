#ifndef MAPSS_H
#define MAPSS_H
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "planet.h"
#include "waypoint.h"
#include "random"
#include <experimental/filesystem>

using namespace std;

#define N_PLANETS 1
#define PLANET_MIN_SIZE 50
#define PLANET_MAX_SIZE 100

#define N_WAYPOINTS 3
#define WAYPOINT_RADIUS 20

#define SHIP_WIDTH WAYPOINT_RADIUS
#define SHIP_HEIGHT WAYPOINT_RADIUS*2

class MapSS
{
 public:
  MapSS();
  MapSS(int size);
  void generate(int nPlanets, int planetMinSize, int planetMaxSize,int nWaypoints, int wpRadius);
  void generateMapPool(int nPlanets, int planetMinSize, int planetMaxSize,int nWaypoints, int wpRadius, string path, int nMaps);
  void save(string filename);
  void load(string filename);
  
  int getSize() const;  
  vector<Planet> getPlanets() const;  
  vector<Waypoint> getWaypoints() const;
  
 private:
  int size;
  vector<Planet> planets;
  vector<Waypoint> waypoints;
};

#endif // MAPSS_H
