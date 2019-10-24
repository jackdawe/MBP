#include "world.h"

World::World(){}

float World::transition()
{
    return 0;
}

bool World::isTerminal(State s)
{
    return false;
}

void World::generateVectorStates() {}

int World::stateId(State s)
{
    return -1;
}

void World::reset()
{

}

vector<int> World::accessibleStates(State s)
{
    return {-1};
}

int World::spaceStateSize()
{
    return -1;
}

void World::saveRewardHistory(string nameTag)
{
    {
        ofstream f(getPath() +"Rewards/"+nameTag);
        if (f)
        {
            for (unsigned int i=0;i<rewardHistory.size();i++)
            {
                f << to_string(rewardHistory[i]) << endl;
            }
        }
        else
        {
            cout << "An error has occured while trying to save the reward history" << endl;
        }
    }
}

void World::saveLastEpisode(string nameTag)
{
    ofstream f("../Sequences/seq" + nameTag);
    if (f)
    {
//        vector<string> paramLabels = agent.getController().getParamLabels();
//        vector<float> paramValues = agent.getController().getParamValues();
//        for (unsigned int i=0; i<paramLabels.size();i++)
//        {
//            f << paramLabels[i] + " = " + to_string(paramValues[i])<<endl;;
//        }
        f << "---SEQUENCE---" <<endl;
        for (unsigned int i=0;i<stateSequence.size();i++)
        {
            string line;
            if (i!=0)
            {
                for (unsigned int j=0;j<actionSequence[0].size();j++)
                {
                    line += to_string(actionSequence[i-1][j]) + " ";
                }
            }
            line += "| ";
            for (unsigned int j=0;j<stateSequence[0].size();j++)

                line+= to_string(stateSequence[i][j]) + " ";
            f << line << endl;
        }
    }
    else
    {
        cout<<"An error has occured when trying to save the sequence"<<endl;
    }
}

void World::loadEpisode(string nameTag)
{
    actionSequence = {};
    stateSequence = {};
    ifstream f("../Sequences/seq" + nameTag);
    if (f)
    {
        string line;
        while (line != "---SEQUENCE---")
        {
            getline(f,line);
        }
        while(getline(f,line))
        {
            int a = line.size();
            int i=0;
            vector<float> vecline;
            while (line[i] != '|')
            {
                string num;
                while(line[i] != ' ')
                {
                    num+=line[i];
                    i++;
                }
                vecline.push_back(stof(num));
                i++;
            }
            i+=2;
            actionSequence.push_back(vecline);
            vecline = vector<float>();
            while (i < line.size())
            {
                string num;
                while(line[i] != ' ')
                {
                    num+=line[i];
                    i++;
                }
                vecline.push_back(stof(num));
                i++;
            }
            stateSequence.push_back(vecline);
        }
    }
    else
    {
        cout<<"An error has occured when trying to load the sequence"<<endl;
    }
}

int World::actionSpaceSize()
{
    return actions.cardinal();
}

int World::saPairSpaceSize()
{
    return spaceStateSize()*actionSpaceSize();
}

ActionSpace World::getActions() const
{
    return actions;
}

State World::getPreviousState() const
{
    return previousState;
}

vector<float> World::getTakenAction() const
{
    return takenAction;
}

float World::getTakenReward() const
{
    return takenReward;
}

State World::getCurrentState() const
{
    return currentState;
}

vector<float> World::getRewardHistory() const
{
    return rewardHistory;
}

void World::addToRewardHistory(float r)
{
    rewardHistory.push_back(r);
}

void World::updateTakenAction(int actionIndex, float value)
{
    takenAction[actionIndex] = value;
}

void World::setActions(const ActionSpace &value)
{
    actions = value;
}

void World::setTakenAction(const vector<float> &value)
{
    takenAction = value;
}

void World::setTakenReward(float value)
{
    takenReward = value;
}

vector<string> World::getParamLabels() const
{
    return paramLabels;
}

vector<float> World::getParamValues() const
{
    return paramValues;
}

string World::getPath() const
{
    return path;
}

vector<vector<float> > World::getStateSequence() const
{
    return stateSequence;
}

vector<vector<float> > World::getActionSequence() const
{
    return actionSequence;
}

int World::getSize() const
{
    return size;
}

