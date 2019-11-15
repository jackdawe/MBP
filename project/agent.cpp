#include "agent.h"

template <class W>
Agent<W>::Agent()
{
}

template <class W>
Agent<W>::Agent(W world, int nEpisodes):
  world(world), episodeNumber(0), nEpisodes(nEpisodes)
{
    this->world.generateVectorStates();
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
    world.saveRewardHistory(nameTag);
}

template<class W>
void Agent<W>::saveLastEpisode()
{
    world.saveLastEpisode(nameTag);
}

template<class W>
void Agent<W>::loadEspisode(string nameTag)
{
    world.loadEpisode(nameTag);
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
    return world.getActions();
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
    return world.getPreviousState();
}

template<class W>
vector<float> Agent<W>::takenAction()
{
    return world.getTakenAction();
}

template<class W>
float Agent<W>::takenReward()
{
    return world.getTakenReward();
}

template<class W>
State Agent<W>::currentState()
{
    return world.getCurrentState();
}

template<class W>
vector<float> Agent<W>::rewardHistory()
{
  return world.getRewardHistory();
}

template<class W>
void Agent<W>::setWorld(const W &value)
{
    world = value;
}

template<class W>
string Agent<W>::getNameTag() const
{
    return nameTag;
}

template<class W>
W Agent<W>::getWorld() const
{
    return world;
}

template class Agent<GridWorld>;
template class Agent<SpaceWorld>;
