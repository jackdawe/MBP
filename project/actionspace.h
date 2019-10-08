#ifndef ACTIONSPACE_H
#define ACTIONSPACE_H
#include "continuousaction.h"
#include "discreteaction.h"

class ActionSpace
{
public:
    ActionSpace();
    ActionSpace(vector<DiscreteAction> discreteActions, vector<ContinuousAction> continuousActions);
    int cardinal();
    int size();
    vector<DiscreteAction> getDiscreteActions() const;

    vector<ContinuousAction> getContinuousActions() const;

private:
    vector<DiscreteAction> discreteActions;
    vector<ContinuousAction> continuousActions;
};

#endif // ACTIONSPACE_H
