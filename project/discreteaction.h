#ifndef DISCRETEACTION_H
#define DISCRETEACTION_H
#include "action.h"

#include <vector>

class DiscreteAction: public Action //A discrete set of actions {0,1,...,size-1}
{
public:
    DiscreteAction();
    DiscreteAction(int size);
    double pick();
private:
    int size;
};

#endif // DISCRETEACTION_H
