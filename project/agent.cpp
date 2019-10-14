#include "agent.h"

template <class C>
Agent<C>::Agent()
{
}

template<class C>
Agent<C>::Agent(C controller, float epsilon): controller(controller), epsilon(epsilon)
{
}

template <class C>
void Agent<C>::epsilonGreedyPolicy()
{
    default_random_engine generator = default_random_engine(random_device{}());
    uniform_real_distribution<float> dist(0,1);
    if (dist(generator) < epsilon)
    {
        for (int i=0;i<daSize();i++)
        {
            controller.updateTakenAction(i,discreteActions()[i].pick());
        }
        for (int i=daSize();i<actions().size();i++)
        {
            controller.updateTakenAction(i,continuousActions()[i].pick());
        }
    }
    else
    {
        greedyPolicy();
    }
    controller.setTakenReward(controller.transition());
}

template <class C>
void Agent<C>::initialiseEpisode()
{
}

template <class C>
void Agent<C>::greedyPolicy()
{
}

template <class C>
void Agent<C>::updatePolicy()
{
}

template <class C>
void Agent<C>::savePolicy(string path)
{
}

template <class C>
void Agent<C>:: loadPolicy(string filename)
{
}

template <class C>
void Agent<C>::generateNameTag(vector<float> parameters, vector<string> parametersName)
{
    string id = "0";
    string tag = "";
    ifstream fr("../idCount");
    if (fr)
    {
        getline(fr,id);
    }
    else
    {
        cout << "an error has occured while trying to read the idCount file" << endl;
    }
    tag+="E";
    string eps = to_string(epsilon);
    tag+=eps[0] + eps[2] + eps[3];
    for (unsigned int i=0;i<parameters.size();i++)
    {
        tag+=parametersName[i];
        string param = to_string(parameters[i]);
        tag+=param[0] + param[2] + param[3];
    }

    ofstream fw("../idCount");
    if (fw)
    {
        fw << to_string(stoi(id)+1) << endl;
    }
    else
    {
        cout << "an error has occured while trying to update the idCount file" << endl;
    }
}

template<class C>
int Agent<C>::daSize()
{
    return actions().getDiscreteActions().size();
}

template<class C>
int Agent<C>::caSize()
{
    return actions().getContinuousActions().size();
}

template<class C>
ActionSpace Agent<C>::actions()
{
    return controller.getActions();
}

template<class C>
vector<DiscreteAction> Agent<C>::discreteActions()
{
    return actions().getDiscreteActions();
}

template<class C>
vector<ContinuousAction> Agent<C>::continuousActions()
{
    return actions().getContinuousActions();
}

template<class C>
State Agent<C>::previousState()
{
    return controller.getPreviousState();
}

template<class C>
vector<float> Agent<C>::takenAction()
{
    return controller.getTakenAction();
}

template<class C>
float Agent<C>::takenReward()
{
    return controller.getTakenReward();
}

template<class C>
State Agent<C>::currentState()
{
    return controller.getCurrentState();
}

template<class C>
vector<float> Agent<C>::rewardHistory()
{
    return controller.getRewardHistory();
}

template<class C>
void Agent<C>::addToRewardHistory(float r)
{
    controller.addToRewardHistory(r);
}

template<class C>
C Agent<C>::getController() const
{
    return controller;
}

template class Agent<ControllerGW>;
