#include "qlearningagent.h"

template <class T>
QLearningAgent<T>::QLearningAgent()
{
}

template <class T>
QLearningAgent<T>::QLearningAgent(vector<Action> actionSpace, float epsilon, float gamma):
    Agent<T>(actionSpace,epsilon), gamma(gamma)
{
    Agent<T>::generateNameTag(vector<float>({gamma}), vector<string>({"G"}));
}

template <class T>
void QLearningAgent<T>::policy()
{
    vector<int> accessibleStates
}

template <class T>
void QLearningAgent<T>::updatePolicy()
{

}

template <class T>
void QLearningAgent<T>::savePolicy()
{

}


template <class T>
void QLearningAgent<T>::loadPolicy()
{

}


template <class T>
void QLearningAgent<T>::saveTrainingData()
{

}
