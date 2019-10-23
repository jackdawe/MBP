#include "controller.h"

Controller::Controller(){}

Controller::Controller(bool imageMode): imageMode(imageMode) {}

float Controller::transition()
{
    return 0;
}

bool Controller::isTerminal(State s)
{
    return false;
}

void Controller::generateVectorStates() {}

void Controller::generateImageStates() {}

int Controller::stateId(State s)
{
    return -1;
}

void Controller::reset()
{

}

vector<int> Controller::accessibleStates(State s)
{
    return {-1};
}

int Controller::spaceStateSize()
{
    return -1;
}

void Controller::saveRewardHistory(string nameTag)
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

void Controller::saveLastEpisode(string nameTag)
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

void Controller::loadEpisode(string nameTag)
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

int Controller::actionSpaceSize()
{
    return actions.cardinal();
}

int Controller::saPairSpaceSize()
{
    return spaceStateSize()*actionSpaceSize();
}

ActionSpace Controller::getActions() const
{
    return actions;
}

State Controller::getPreviousState() const
{
    return previousState;
}

vector<float> Controller::getTakenAction() const
{
    return takenAction;
}

float Controller::getTakenReward() const
{
    return takenReward;
}

State Controller::getCurrentState() const
{
    return currentState;
}

vector<float> Controller::getRewardHistory() const
{
    return rewardHistory;
}

void Controller::addToRewardHistory(float r)
{
    rewardHistory.push_back(r);
}

void Controller::updateTakenAction(int actionIndex, float value)
{
    takenAction[actionIndex] = value;
}

void Controller::setActions(const ActionSpace &value)
{
    actions = value;
}

void Controller::setTakenAction(const vector<float> &value)
{
    takenAction = value;
}

void Controller::setTakenReward(float value)
{
    takenReward = value;
}

vector<string> Controller::getParamLabels() const
{
    return paramLabels;
}

vector<float> Controller::getParamValues() const
{
    return paramValues;
}

string Controller::getPath() const
{
    return path;
}

vector<vector<float> > Controller::getStateSequence() const
{
    return stateSequence;
}

vector<vector<float> > Controller::getActionSequence() const
{
    return actionSequence;
}

bool Controller::getImageMode() const
{
    return imageMode;
}

