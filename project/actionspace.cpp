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
        int cardinal;
        for (unsigned i=0;i<discreteActions.size();i++)
        {
            cardinal *= discreteActions[i].getSize();
        }
        return discreteActions.size();
    }
}

int ActionSpace::size()
{
    return discreteActions.size() + continuousActions.size();
}

vector<double> ActionSpace::actionFromId(int id, vector<double> *p_coordinates, unsigned int counter)
{
    if (counter != discreteActions.size())
    {
        counter++;
        int product = 1;
        for (unsigned int i=0;i<discreteActions.size()-counter;i++)
        {
            product*=discreteActions[i].getSize();
        }
        p_coordinates->push_back(id/product);
        actionFromId(id-((id/product)*product),p_coordinates,counter);
    }
    vector<double> coordinates = *p_coordinates;
    return coordinates;
}

vector<DiscreteAction> ActionSpace::getDiscreteActions() const
{
    return discreteActions;
}

vector<ContinuousAction> ActionSpace::getContinuousActions() const
{
    return continuousActions;
}
