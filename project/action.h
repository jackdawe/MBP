#ifndef ACTION_H
#define ACTION_H
#include <stdio.h>
#include <random>
using namespace std;

class Action
{
public:
    Action();
    virtual double pick();
};

#endif // ACTION_H
