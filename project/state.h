#ifndef STATE_H
#define STATE_H
#include "action.h"

class State
{
public:
    State();
    void add(double* value);

    vector<double *> getStateVector() const;
    void setStateVector(const vector<double *> &value);

protected:
    vector<double*> stateVector;
};

#endif // STATE_H
