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
    MapGW(int size, string mapTag);
    void generate(int obstacleMaxNumber);
    void save();
    void load(string mapTag);
    int getSize() const;

    string getMapTag() const;

    vector<vector<int> > getMap() const;

private:
    int size;
    vector<vector<int>> map;
    string mapTag;

};

#endif // MAPGW_H
