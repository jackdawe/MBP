#ifndef DISCRETEACTION_H
#define DISCRETEACTION_H
#include "action.h"

#include <vector>

class DiscreteAction: public Action //A discrete set of actions {0,1,...,size-1}
{
public:
    DiscreteAction();
    DiscreteAction(int size);
    float pick();
    int getSize() const;

private:
    int size;
};

#endif // DISCRETEACTION_H
