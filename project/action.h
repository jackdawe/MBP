#ifndef ACTION_H
#define ACTION_H
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

class Action
{
public:
    Action();
    virtual float pick();
};

#endif // ACTION_H
