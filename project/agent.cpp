#include "agent.h"

template <class W>
Agent<W>::Agent()
{
}

template <class W>
Agent<W>::Agent(W controller):
    controller(controller), episodeNumber(0)
{
    this->controller.generateVectorStates();
}

template <class W>
void Agent<W>::generateNameTag(string prefix)
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

template<class W>
void Agent<W>::saveRewardHistory()
{
    controller.saveRewardHistory(nameTag);
}

template<class W>
void Agent<W>::saveLastEpisode()
{
    controller.saveLastEpisode(nameTag);
}

template<class W>
void Agent<W>::loadEspisode(string nameTag)
{
    controller.loadEpisode(nameTag);
}

template<class W>
int Agent<W>::daSize()
{
    return actions().getDiscreteActions().size();
}

template<class W>
int Agent<W>::caSize()
{
    return actions().getContinuousActions().size();
}

template<class W>
ActionSpace Agent<W>::actions()
{
    return controller.getActions();
}

template<class W>
vector<DiscreteAction> Agent<W>::discreteActions()
{
    return actions().getDiscreteActions();
}

template<class W>
vector<ContinuousAction> Agent<W>::continuousActions()
{
    return actions().getContinuousActions();
}

template<class W>
State Agent<W>::previousState()
{
    return controller.getPreviousState();
}

template<class W>
vector<float> Agent<W>::takenAction()
{
    return controller.getTakenAction();
}

template<class W>
float Agent<W>::takenReward()
{
    return controller.getTakenReward();
}

template<class W>
State Agent<W>::currentState()
{
    return controller.getCurrentState();
}

template<class W>
vector<float> Agent<W>::rewardHistory()
{
  return controller.getRewardHistory();
}

template<class W>
void Agent<W>::setController(const W &value)
{
    controller = value;
}

template<class W>
string Agent<W>::getNameTag() const
{
    return nameTag;
}

template<class W>
W Agent<W>::getController() const
{
    return controller;
}

template class Agent<GridWorld>;
template class Agent<SpaceWorld>;
