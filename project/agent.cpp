#include "agent.h"

template <class T>
Agent<T>::Agent()
{
}

template<class T>
Agent<T>::Agent(vector<Action> actionSpace, float epsilon): actionSpace(actionSpace), epsilon(epsilon)
{
}

template <class T>
void Agent<T>::epsilonGreedyPolicy()
{
    default_random_engine generator = default_random_engine(random_device{}());
    uniform_real_distribution<float> dist(0,1);
    if (dist(generator) < epsilon)
    {
        for (int i=0;i<actionSpace.size();i++)
        {
            takenAction[i] = actionSpace[i].pick();
        }
    }
    else
    {
        policy();
    }
}

template <class T>
void Agent<T>::initialiseEpisode()
{
}

template <class T>
void Agent<T>::policy()
{
}

template <class T>
void Agent<T>::updatePolicy()
{
}

template <class T>
void Agent<T>::savePolicy()
{
}

template <class T>
void Agent<T>:: loadPolicy()
{
}

template <class T>
void Agent<T>::saveTrainingData()
{
}

template <class T>
void Agent<T>::addToRewardHistory(double r)
{
    rewardHistory.push_back(r);
}

template <class T>
void Agent<T>::generateNameTag(vector<float> parameters, vector<string> parametersName)
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
