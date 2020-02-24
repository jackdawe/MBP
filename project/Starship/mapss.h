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

#define SIZE 800 //You shouldn't change this too much

#define N_PLANETS 1
#define PLANET_MIN_SIZE 50
#define PLANET_MAX_SIZE 100

#define N_WAYPOINTS 3
#define WAYPOINT_RADIUS 15

class MapSS
{
 public:
  MapSS();
  MapSS(vector<float> stateVector, int nWaypoints);
  void generate(int nPlanets, int planetMinSize, int planetMaxSize,int nWaypoints, int wpRadius);
  void generateMapPool(int nPlanets, int planetMinSize, int planetMaxSize,int nWaypoints, int wpRadius, string path, int nMaps);
  void save(string filename);
  void load(string filename);
  
  vector<Planet> getPlanets() const;  
  vector<Waypoint> getWaypoints() const;
  
 private:
  vector<Planet> planets;
  vector<Waypoint> waypoints;
};

#endif // MAPSS_H
