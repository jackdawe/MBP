#include "agent.h"

template <class C>
Agent<C>::Agent()
{
}

template<class C>
Agent<C>::Agent(C controller, float epsilon): controller(controller), epsilon(epsilon), episodeNumber(0)
{
    this->controller.generateStates();
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
        for (int i=0;i<caSize();i++)
        {
            controller.updateTakenAction(i+daSize(),continuousActions()[i].pick());
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
void Agent<C>::finaliseEpisode()
{
}

template <class C>
void Agent<C>::savePolicy()
{
}

template <class C>
void Agent<C>:: loadPolicy(string tag)
{
}

template <class C>
void Agent<C>::generateNameTag(vector<float> parameters, vector<string> parametersName)
{
    string id;
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
    tag+=eps[0],tag+=eps[2],tag+=eps[3];
    for (unsigned int i=0;i<parameters.size();i++)
    {
        tag+=parametersName[i];
        string param = to_string(parameters[i]);
        tag+=param[0],tag+=param[2],tag+=param[3];
    }
    nameTag = tag+"_"+id;
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
void Agent<C>::resetController()
{
    controller.reset();
}

template<class C>
void Agent<C>::incrementEpisode()
{
    episodeNumber++;
}

template<class C>
void Agent<C>::saveRewardHistory()
{
    {
        ofstream f(controller.getPath() +"Rewards/"+this->nameTag);
        if (f)
        {
            for (unsigned int i=0;i<rewardHistory().size();i++)
            {
                f << to_string(rewardHistory()[i]) << endl;
            }
        }
        else
        {
            cout << "An error has occured while trying to save the reward history" << endl;
        }
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
void Agent<C>::setController(const C &value)
{
    controller = value;
}

template<class C>
C Agent<C>::getController() const
{
    return controller;
}

template class Agent<ControllerGW>;
template class Agent<ControllerSS>;
