#include "qlearning.h"

template <class T>
QLearning<T>::QLearning()
{
}

template <class T>
QLearning<T>::QLearning(vector<Action> actionSpace, float epsilon, float gamma):
    Agent<T>(actionSpace,epsilon), gamma(gamma)
{
    Agent<T>::generateNameTag(vector<float>({gamma}), vector<string>({"G"}));
}

template <class T>
void QLearning<T>::greedyPolicy()
{
//    vector<int> accessibleStates = currentState.accessibleStates();
//    vector<double> possibleQValues;
//    for (unsigned int i=0;i<accessibleStates.size;i++)
//    {
//        possibleQValues.push_back(qvalues[]);
//    }
}

template <class T>
void QLearning<T>::updatePolicy()
{

}

template <class T>
void QLearning<T>::savePolicy()
{

}


template <class T>
void QLearning<T>::loadPolicy()
{

}


template <class T>
void QLearning<T>::saveTrainingData()
{

}
