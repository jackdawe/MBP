#include "agent.h"

template <class C>
Agent<C>::Agent()
{
}

template <class C>
Agent<C>::Agent(C controller):
    controller(controller), episodeNumber(0)
{
    if(this->controller.getImageMode())
    {
        this->controller.generateImageStates();
    }
    else
    {
       this->controller.generateVectorStates();
    }
}

template <class C>
void Agent<C>::generateNameTag(string prefix)
{
    string id;
    ifstream fr("../idCount");
    if (fr)
    {
        getline(fr,id);
    }
    else
    {
        cout << "an error has occured while trying to read the idCount file" << endl;
    }
    nameTag = prefix +"_"+id;
    ofstream fw("../idCount");
    if (fw)
    {
        fw << to_string(stoi(id)+1) << endl;
    }
    else
    {
        cout << "an error has occured while trying to update the idCount file" << endl;
    }
    cout << "Files saved during training under the name: " + nameTag << endl;
}

template<class C>
void Agent<C>::incrementEpisode()
{
    episodeNumber++;
}

template<class C>
void Agent<C>::saveRewardHistory()
{
    controller.saveRewardHistory(nameTag);
}

template<class C>
void Agent<C>::saveLastEpisode()
{
    controller.saveLastEpisode(nameTag);
}

template<class C>
void Agent<C>::loadEspisode(string nameTag)
{
    controller.loadEpisode(nameTag);
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

//template<class C>
//vector<float> Agent<C>::rewardHistory()
//{
//    return controller.getRewardHistory();
//}

//template<class C>
//void Agent<C>::addToRewardHistory(float r)
//{
//    controller.addToRewardHistory(r);
//}

template<class C>
void Agent<C>::setController(const C &value)
{
    controller = value;
}

template<class C>
string Agent<C>::getNameTag() const
{
    return nameTag;
}

template<class C>
C Agent<C>::getController() const
{
    return controller;
}

template class Agent<ControllerGW>;
template class Agent<ControllerSS>;
