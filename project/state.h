#ifndef STATE_H
#define STATE_H
#include "action.h"

class State
{
public:
    State();

protected:
    vector<double*> stateVector;
};

#endif // STATE_H
