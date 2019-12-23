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
  unsigned int cardinal=1;
  for (unsigned int i=0;i<discreteActions.size();i++)
    {
      cardinal *= discreteActions[i].getSize();
      }
  return cardinal;    
}

int ActionSpace::nActions()
{
  unsigned int nAct=0;
  for (unsigned int i=0;i<discreteActions.size();i++)
    {
      nAct += discreteActions[i].getSize();
    }
  return nAct+continuousActions.size();
}
  
int ActionSpace::size()
{
  return discreteActions.size() + continuousActions.size();
}

vector<float> ActionSpace::actionFromId(int id, vector<float> *p_coordinates, unsigned int counter)
{
    if (counter != discreteActions.size())
    {
        counter++;
        int product = 1;
        for (unsigned int i=counter;i<discreteActions.size();i++)
        {
            product*=discreteActions[i].getSize();
        }
        p_coordinates->push_back(id/product);
        actionFromId(id-((id/product)*product),p_coordinates,counter);
    }
    vector<float> coordinates = *p_coordinates;
    return coordinates;
}

int ActionSpace::idFromAction(vector<float> actions)
{
    int id=0;
    for (unsigned int i=0;i<discreteActions.size();i++)
    {
        int product = 1;
        for (unsigned int j=i+1;j<discreteActions.size();j++)
        {
            product*=discreteActions[j].getSize();
        }
        id+=product*actions[i];
    }
    return id;
}

vector<DiscreteAction> ActionSpace::getDiscreteActions() const
{
    return discreteActions;
}

vector<ContinuousAction> ActionSpace::getContinuousActions() const
{
    return continuousActions;
}
