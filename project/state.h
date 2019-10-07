#ifndef STATE_H
#define STATE_H
#include "action.h"

class State
{
public:
    State();
    virtual void transition(double a); //Changing the state after performing a
    virtual double reward(double a); //Returns the reward for taking action a
    virtual bool isTerminal(); //returns true if the state is terminal
    virtual void generateStateVector(); //Converts any representation of a state to a state vector readable by the agentTrainer
protected:
    vector<double*> stateVector;
};

#endif // STATE_H
