#ifndef ACTIONSPACE_H
#define ACTIONSPACE_H
#include "continuousaction.h"
#include "discreteaction.h"

/**
 * @brief ActionSpace
 * Class to represent the action space of a world. For example, in the space world, the space ship has an action space containing one discrete action and two contiunous actions.
 */
class ActionSpace
{
public:
    ActionSpace();
    ActionSpace(vector<DiscreteAction> discreteActions, vector<ContinuousAction> continuousActions);
    
    /**
     * @brief cardinal
     * Computes the cardinal of the discrete action space, i.e. the number of possible actions.  
     * @return the cardinal of the discrete action space. 
     */
    int cardinal();

    /**
     * @brief nActions
     * Computes the total number of elementary actions of the action space. 
     * @return the sum of the sizes of the discrete actions and of the number of continuous actions. 
     */    
    int nActions();
    
    /**
     *
     * @brief size 
     * Computes the length of the action vector
     * @return the length of the the action vector
     */
    int size();

    /**
     *
     * @brief actionFromId
     * This method only works if cardinal of the action space is finite.
     * Transforms an element from [0 nA1*nA2*..nAn] into an element of [0 nA1]x[0 nA2]x...x[0 nAn].
     * THe purpose is to convert an integer representation of an action vector into the said action vector.
     * @param id the element to transform 
     * @param p_coordinates an empty vector of pointers that will contain the future coordinates 
     * @param counter a counter for reccursion
     * @return the coordinates of id in the new space 
     */
    vector<float> actionFromId(int id, vector<float> *p_coordinates, unsigned int counter = 0);

    /**
     * @brief idFromAction
     * inverse of the actionFromId function.
     * Converts an action vector into its integer representation
     * @param  actions a vector
     * @return the integer representation of actions
     */
    int idFromAction(vector<float> actions);

    vector<DiscreteAction> getDiscreteActions() const;
    vector<ContinuousAction> getContinuousActions() const;

private:
    /** A vector containing the discrete actions 
     */
    vector<DiscreteAction> discreteActions;
    /** A vector containing the continuous actions
     */
    vector<ContinuousAction> continuousActions;
};

#endif // ACTIONSPACE_H
