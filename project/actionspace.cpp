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

int ActionSpace::size()
{
    return discreteActions.size() + continuousActions.size();
}

vector<DiscreteAction> ActionSpace::getDiscreteActions() const
{
    return discreteActions;
}

vector<ContinuousAction> ActionSpace::getContinuousActions() const
{
    return continuousActions;
}
