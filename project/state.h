#ifndef STATE_H
#define STATE_H
#include "action.h"

class State
{
public:
    State();
    virtual void transition(Action a); //Changing the state after performing a
    virtual double reward(Action a); //Returns the reward for taking action a
    virtual bool isTerminal(); //returns true if the state is terminal
};

#endif // STATE_H
