#ifndef ACTION_H
#define ACTION_H
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

/**
 * @brief Class representing any type of action
 */
class Action
{
public:
    Action();
    /**
     * @brief pick
     * picks a random action 
     * @return the value of the action
     */ 
    virtual float pick();
};

#endif // ACTION_H
