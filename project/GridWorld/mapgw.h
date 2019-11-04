#ifndef MAPGW_H
#define MAPGW_H
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

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
