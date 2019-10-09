#include "qlearning.h"

template <class C>
QLearning<C>::QLearning()
{
}

template <class C>
QLearning<C>::QLearning(vector<Action> actionSpace, float epsilon, float gamma):
    Agent<C>(actionSpace,epsilon), gamma(gamma)
{
    Agent<C>::generateNameTag(vector<float>({gamma}), vector<string>({"G"}));

    //Initialising the Q fonction to 0 for each state action pair

    for (int i=0;i<Agent<C>::controller.spaceStateSize()*Agent<C>::actions().cardinal() ;i++)
    {
        qvalues[i] = 0;
    }
}

template <class C>
void QLearning<C>::greedyPolicy()
{
    vector<int> accessibleStates = this->controller.accessibleStates();
    vector<double> possibleQValues;

    for (unsigned i=0;i<Agent<C>::actions().cardinal();i++)
    {
        possibleQValues.push_back(qvalues[this->controller.stateId(this->currentState())*this->actions().cardinal()+i]);
    }
    double maxQValue = *max_element(possibleQValues.begin(),possibleQValues.end());

    //If several qvalues are equal to the max, pick one of them randomly

    vector<int> indexOfMax;
    for (unsigned int i=0;i<possibleQValues.size();i++)
    {
        if (possibleQValues[i]==maxQValue)
        {
            indexOfMax.push_back(i);
        }
    }
    default_random_engine generator(random_device{}());
    uniform_int_distribution<int> dist(0,indexOfMax.size()-1);
    int actionId = indexOfMax[dist(generator)];
    this->controller.setTakenAction(this->actions().actionFromId(actionId,new vector<double>()));
}

template <class C>
void QLearning<C>::updatePolicy()
{

}

template <class C>
void QLearning<C>::savePolicy()
{

}


template <class C>
void QLearning<C>::loadPolicy()
{

}


template <class C>
void QLearning<C>::saveTrainingData()
{

}
