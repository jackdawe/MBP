#ifndef MAPGW_H
#define MAPGW_H
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <gflags/gflags.h>
DECLARE_int32(size);
DECLARE_int32(maxObst);
DECLARE_string(f);
DECLARE_string(dir);
DECLARE_int32(nmaps);
 
 using namespace std;

class MapGW
{

public:
    MapGW();
    MapGW(int size);
    void generate(int obstacleMaxNumber);
    void generateMapPool(int obstacleMaxNumber, string path, int nMaps);
    void save(string filename);
    void load(string filename);
    int getSize() const;
    vector<vector<int> > getMap() const;

private:
    int size;
    vector<vector<int>> map;

};

#endif // MAPGW_H
