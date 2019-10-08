#include "actionspace.h"

ActionSpace::ActionSpace()
{
}

ActionSpace::ActionSpace(vector<DiscreteAction> discreteActions, vector<ContinuousAction> continuousActions):
    discreteActions(discreteActions), continuousActions(continuousActions)
{
}

int ActionSpace::cardinal()
{
    if (continuousActions.size() != 0)
    {
        return -1;
    }
    else
    {
        return discreteActions.size();
    }
}
